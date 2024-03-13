
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
.text
.protected CustomGSUs_Cijk_Ailk_Bljk_HHS_BH_Bias_AS_SAV_MT32x16x16_MI16x16x1_prefetch
.globl CustomGSUs_Cijk_Ailk_Bljk_HHS_BH_Bias_AS_SAV_MT32x16x16_MI16x16x1_prefetch
.p2align 8
.type CustomGSUs_Cijk_Ailk_Bljk_HHS_BH_Bias_AS_SAV_MT32x16x16_MI16x16x1_prefetch,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel CustomGSUs_Cijk_Ailk_Bljk_HHS_BH_Bias_AS_SAV_MT32x16x16_MI16x16x1_prefetch
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_accum_offset 64 // accvgpr offset
  .amdhsa_next_free_vgpr 68 // vgprs
  .amdhsa_next_free_sgpr 66 // sgprs
  .amdhsa_group_segment_fixed_size 1824 // lds bytes
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel
.text
/* Num VGPR   =60 */
/* Num AccVGPR=4 */
/* Num SGPR   =66 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 4 x 1 */
/* SubGroup= 8 x 16 */
/* VectorWidthA=1 */
/* VectorWidthB=1 */
/* GlobalReadVectorWidthA=4, GlobalReadVectorWidthB=2 */
/* DirectToLdsA=False */
/* DirectToLdsB=False */
/* UseSgprForGRO=1 */
.amdgpu_metadata
---
custom.config:
   InternalSupportParams:
      SupportUserGSU: True
      SupportCustomWGM: True
      SupportCustomStaggerU: False
      UseUniversalArgs: False
amdhsa.version:
  - 1
  - 1
amdhsa.kernels:
  - .name: CustomGSUs_Cijk_Ailk_Bljk_HHS_BH_Bias_AS_SAV_MT32x16x16_MI16x16x1_prefetch
    .symbol: 'CustomGSUs_Cijk_Ailk_Bljk_HHS_BH_Bias_AS_SAV_MT32x16x16_MI16x16x1_prefetch.kd'
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
      - .name:            AddressScaleAlphaVec
        .size:            8
        .offset:          92
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            bias
        .size:            8
        .offset:          100
        .value_kind:      global_buffer
        .value_type:      void
        .address_space:   generic
      - .name:            biasType
        .size:            4
        .offset:          108
        .value_kind:      by_value
        .value_type:      u32
      - .name:            StrideBias
        .size:            4
        .offset:          112
        .value_kind:      by_value
        .value_type:      u32
      - .name:            activationAlpha
        .size:            4
        .offset:          116
        .value_kind:      by_value
        .value_type:      f32
      - .name:            activationBeta
        .size:            4
        .offset:          120
        .value_kind:      by_value
        .value_type:      f32
      - .name:            activationType
        .size:            4
        .offset:          124
        .value_kind:      by_value
        .value_type:      u32
    .group_segment_fixed_size:   1824
    .kernarg_segment_align:      8
    .kernarg_segment_size:       128
    .max_flat_workgroup_size:    128
    .private_segment_fixed_size: 0
    .sgpr_count:                 66
    .sgpr_spill_count:           0
    .vgpr_count:                 60
    .vgpr_spill_count:           0
    .wavefront_size:             64
...
.end_amdgpu_metadata
CustomGSUs_Cijk_Ailk_Bljk_HHS_BH_Bias_AS_SAV_MT32x16x16_MI16x16x1_prefetch:

.long 0xC0120400, 0x00000000
.long 0xC00A0800, 0x00000040
.long 0xC0060900, 0x00000050
.long 0xC0020980, 0x00000058
.long 0xBEFC00FF, 0x00000720
.long 0x7E180300
.long 0xBEBC1C00
.long 0xBEBE00C1
.long 0xBEBF00FF, 0x00020000
.long 0xD1FD003E, 0x02010F0C
.long 0xE0501000, 0x800F3F3E
.long 0xD1FD003E, 0x04F90F0C
.long 0xE0501000, 0x800F3F3E
.long 0x261C18BF
.long 0x261A1C8F
.long 0x261C18BF
.long 0x201C1C84
.long 0x241C1C87
.long 0x681A1B0E
.long 0x201C1886
.long 0x261C1C81
.long 0x241C1C84
.long 0x681A1B0E
.long 0x261E18BF
.long 0x261C1E8F
.long 0x241C1C84
.long 0x261E18BF
.long 0x201E1E84
.long 0x241E1E82
.long 0x681C1D0F
.long 0x201E1886
.long 0x201E1E81
.long 0xBEB900FF, 0x00000200
.long 0xD285000F, 0x00021E39
.long 0xD1FE000A, 0x02061B0F
.long 0x20201488
.long 0x24202085
.long 0x68141510
.long 0x201A1886
.long 0x201A1A81
.long 0xBEB90090
.long 0xD285000D, 0x00021A39
.long 0xD1FE000B, 0x02061D0D
.long 0x201E1687
.long 0x241E1E83
.long 0x6816170F
.long 0x321616FF, 0x00000500
.long 0x201C1883
.long 0x261A1887
.long 0x241A1A82
.long 0x7E1E030E
.long 0x20201883
.long 0x26221887
.long 0x24222281
.long 0x7E280311
.long 0x100C1EA0
.long 0xD1FE0006, 0x02060D0D
.long 0x20240C88
.long 0x24242485
.long 0x680C0D12
.long 0x100E2090
.long 0xD1FE0007, 0x02060F14
.long 0x20240E87
.long 0x24242483
.long 0x680E0F12
.long 0x320E0EFF, 0x00000500
.long 0xBF8CC07F
.long 0x8609FF26, 0x0000FF00
.long 0x8F098809
.long 0x8627FF26, 0xFFFF0000
.long 0x8F279027
.long 0x8626FF26, 0x000000FF
.long 0x7E2E02A0
.long 0x7E2C0210
.long 0x7E2A0D17
.long 0x7E2A4715
.long 0x7E300D16
.long 0x0A2A3115
.long 0x7E2A0F15
.long 0x10302F15
.long 0x6A303116
.long 0xD0CD006A, 0x00010118
.long 0xD11C6A15, 0x01A90115
.long 0x7E2E0290
.long 0x7E2C0211
.long 0x7E180515
.long 0x7E2A0D17
.long 0x7E2A4715
.long 0x7E300D16
.long 0x0A2A3115
.long 0x7E2A0F15
.long 0x10302F15
.long 0x6A303116
.long 0xD0CD006A, 0x00010118
.long 0xD11C6A15, 0x01A90115
.long 0xBF800000
.long 0x7E1A0515
.long 0x80988818
.long 0x82998019
.long 0x809A841A
.long 0x829B801B
.long 0xD042006A, 0x00010024
.long 0xBF860001
.long 0xBE930080
.long 0xBF068126
.long 0xBF850012
.long 0x7E240C26
.long 0x7E244712
.long 0x7E260C03
.long 0x0A242712
.long 0x7E240F12
.long 0xD1080013, 0x00004D12
.long 0x6A262603
.long 0xD0DA007E, 0x00004D13
.long 0x68242481
.long 0x7E260280
.long 0xBEFE01C1
.long 0x7E060512
.long 0x7E0C0513
.long 0xBE850081
.long 0xBE880082
.long 0xBF820003
.long 0xBE860180
.long 0xBE850081
.long 0xBE880081
.long 0xBF0B8109
.long 0xBF850035
.long 0x7E240C09
.long 0x7E244712
.long 0x7E260C03
.long 0x0A242712
.long 0x7E240F12
.long 0xD1080013, 0x00001312
.long 0x6A262603
.long 0xD0DA007E, 0x00001313
.long 0x68242481
.long 0xBEFE01C1
.long 0x7E780512
.long 0x9239093C
.long 0x80B93903
.long 0x92390C39
.long 0x80390239
.long 0x7E240C09
.long 0x7E244712
.long 0x7E260C0D
.long 0x0A242712
.long 0x7E240F12
.long 0xD1080013, 0x00001312
.long 0x6A26260D
.long 0xD0DA007E, 0x00001313
.long 0x68242481
.long 0xBEFE01C1
.long 0x7E740512
.long 0x923B3A09
.long 0x80BB3B0D
.long 0xBF06803B
.long 0xBEBB0209
.long 0xBF093A3C
.long 0x853A093B
.long 0x7E240C3A
.long 0x7E244712
.long 0x7E260C39
.long 0x0A242712
.long 0x7E240F12
.long 0xD1080013, 0x00007512
.long 0x6A262639
.long 0xD0DA007E, 0x00007513
.long 0x68242481
.long 0x7E260280
.long 0xBEFE01C1
.long 0x7E040512
.long 0x7E060513
.long 0x923C093C
.long 0x80033C03
.long 0xD2850015, 0x00021C20
.long 0x32102B0D
.long 0x68101084
.long 0x24101081
.long 0xD285000D, 0x00022022
.long 0x32121B11
.long 0x68121282
.long 0x24121281
.long 0x963DA002
.long 0x923CA002
.long 0x963B0690
.long 0x923A0690
.long 0x963B203A
.long 0x923A203A
.long 0x803C3A3C
.long 0x823D3B3D
.long 0xBE8E0081
.long 0xBE8F0080
.long 0x80BA8110
.long 0x963B3A81
.long 0x923A3A81
.long 0x800E3A0E
.long 0x820F3B0F
.long 0x80BA8113
.long 0x963B3A20
.long 0x923A3A20
.long 0x800E3A0E
.long 0x820F3B0F
.long 0x808E3C0E
.long 0x828F3D0F
.long 0x8E8E810E
.long 0x800E880E
.long 0x820F800F
.long 0xBF06800F
.long 0x852AC10E
.long 0x963B0421
.long 0x923A0421
.long 0x803C3A3C
.long 0x823D3B3D
.long 0x8EBC813C
.long 0x80283C18
.long 0x82293D19
.long 0xBEAB00FF, 0x00020000
.long 0x963D9003
.long 0x923C9003
.long 0x963D223C
.long 0x923C223C
.long 0x963B0690
.long 0x923A0690
.long 0x803C3A3C
.long 0x823D3B3D
.long 0xBEB00081
.long 0xBEB10080
.long 0x80BA8113
.long 0x963B3A81
.long 0x923A3A81
.long 0x80303A30
.long 0x82313B31
.long 0x80BA8111
.long 0x963B3A22
.long 0x923A3A22
.long 0x80303A30
.long 0x82313B31
.long 0x80B03C30
.long 0x82B13D31
.long 0x8EB08130
.long 0x80308430
.long 0x82318031
.long 0xBF068031
.long 0x852EC130
.long 0x963B0423
.long 0x923A0423
.long 0x803C3A3C
.long 0x823D3B3D
.long 0x8EBC813C
.long 0x802C3C1A
.long 0x822D3D1B
.long 0xBEAF00FF, 0x00020000
.long 0x9239A026
.long 0x92372039
.long 0x9239A026
.long 0xBEB80039
.long 0xD3D94000, 0x18000080
.long 0xD3D94001, 0x18000080
.long 0xD3D94002, 0x18000080
.long 0xD3D94003, 0x18000080
.long 0x8F0A8413
.long 0xBF068126
.long 0xBF850012
.long 0x7E1A0C26
.long 0x7E1A470D
.long 0x7E1C0C0A
.long 0x0A1A1D0D
.long 0x7E1A0F0D
.long 0xD108000E, 0x00004D0D
.long 0x6A1C1C0A
.long 0xD0DA007E, 0x00004D0E
.long 0x681A1A81
.long 0x7E1C0280
.long 0xBEFE01C1
.long 0x7E14050D
.long 0x7E0E050E
.long 0x803A0A81
.long 0xBF0A0706
.long 0xBE8A023A
.long 0xBE8B000A
.long 0x863CFF27, 0x00001F00
.long 0x8F3C883C
.long 0x863DFF27, 0x0000E000
.long 0x8627FF27, 0x000000FF
.long 0xBEBA0027
.long 0x8E3B3C3A
.long 0xBF093B0B
.long 0xBF850002
.long 0x8F3A813A
.long 0xBF82FFFB
.long 0x80BB813A
.long 0xBF09813A
.long 0x8532803B
.long 0xBF06803D
.long 0xBF850002
.long 0xBEBA0002
.long 0xBF820016
.long 0xBF06FF3D, 0x00002000
.long 0xBF850002
.long 0xBEBA0003
.long 0xBF820011
.long 0xBF06FF3D, 0x00004000
.long 0xBF850002
.long 0xBEBA00C1
.long 0xBF82000C
.long 0xBF06FF3D, 0x00006000
.long 0xBF850004
.long 0x923B030C
.long 0x803A3B3A
.long 0x803A023A
.long 0xBF820005
.long 0xBF06FF3D, 0x00008000
.long 0xBF850002
.long 0xBEBA00C1
.long 0xBF820000
.long 0x86323A32
.long 0x8E323C32
.long 0x96BB3732
.long 0x923A3732
.long 0x96B4370A
.long 0x9233370A
.long 0x80B33337
.long 0x82B43480
.long 0x80283A28
.long 0x82293B29
.long 0x808E3A0E
.long 0x828F3B0F
.long 0xBF06800F
.long 0x852AC10E
.long 0x96BB3832
.long 0x923A3832
.long 0x96B6380A
.long 0x9235380A
.long 0x80B53538
.long 0x82B63680
.long 0x802C3A2C
.long 0x822D3B2D
.long 0x80B03A30
.long 0x82B13B31
.long 0xBF068031
.long 0x852EC130
.long 0x80328132
.long 0xBF0B800A
.long 0xBF850034
.long 0xE0541000, 0x800A0008
.long 0xE0501000, 0x800B0409
.long 0xBF06320A
.long 0x853A3733
.long 0x853B8034
.long 0x80283A28
.long 0x82293B29
.long 0x808E3A0E
.long 0x828F3B0F
.long 0xBF06800F
.long 0x852AC10E
.long 0xBF06320A
.long 0x853A3835
.long 0x853B8036
.long 0x802C3A2C
.long 0x822D3B2D
.long 0x80B03A30
.long 0x82B13B31
.long 0xBF068031
.long 0x852EC130
.long 0xBF8C0F70
.long 0xBF8C0000
.long 0xBF8A0000
.long 0xD89A0000, 0x00000006
.long 0xD81A0000, 0x00000407
.long 0xBF8CC07F
.long 0xBF8C0000
.long 0xBF8A0000
.long 0xD8780000, 0x0000000A
.long 0xD8B60040, 0x0200000A
.long 0xD8780080, 0x0100000A
.long 0xD8B600C0, 0x0300000A
.long 0xD8EC0000, 0x0400000B
.long 0xBF8CC07F
.long 0x28000500
.long 0x28020701
.long 0xBF800000
.long 0xBF800000
.long 0xD3CD8000, 0x04020900
.long 0x808A810A
.long 0xBF00800A
.long 0xBF84FFCC
.long 0xBEBC1C00
.long 0xBEBE00C1
.long 0xBEBF00FF, 0x00020000
.long 0xD1FD003E, 0x02010F0C
.long 0xE0501000, 0x800F3F3E
.long 0xD1FD003E, 0x04F90F0C
.long 0xE0501000, 0x800F3F3E
.long 0x860A138F
.long 0xBF070706
.long 0xBE8A0280
.long 0xBF06800A
.long 0xBE8B0080
.long 0xBF85006A
.long 0x81BA3282
.long 0x96BB373A
.long 0x923A373A
.long 0x80BA333A
.long 0x82BB343B
.long 0x80283A28
.long 0x82293B29
.long 0x808E3A0E
.long 0x828F3B0F
.long 0xBF06800F
.long 0x852AC10E
.long 0x81BA3282
.long 0x96BB383A
.long 0x923A383A
.long 0x80BA353A
.long 0x82BB363B
.long 0x802C3A2C
.long 0x822D3B2D
.long 0x80B03A30
.long 0x82B13B31
.long 0xBF068031
.long 0x852EC130
.long 0xE0901000, 0x800A0008
.long 0xE0941002, 0x800A0D08
.long 0xBF8C0F70
.long 0x28001B00
.long 0xE0901004, 0x800A0108
.long 0xE0941006, 0x800A0D08
.long 0xBF8C0F70
.long 0x28021B01
.long 0xE0901000, 0x800B0409
.long 0xE0941002, 0x800B0D09
.long 0xBF8C0F70
.long 0x28081B04
.long 0xBF8C0F70
.long 0xBF8C0000
.long 0xBF8A0000
.long 0xD89A0000, 0x00000006
.long 0xD81A0000, 0x00000407
.long 0xBF8CC07F
.long 0xBF8C0000
.long 0xBF8A0000
.long 0xD8780000, 0x0000000A
.long 0xD8B60040, 0x0200000A
.long 0xD8780080, 0x0100000A
.long 0xD8B600C0, 0x0300000A
.long 0xD8EC0000, 0x0400000B
.long 0xBEB900FF, 0x00000480
.long 0x32141439
.long 0xBEB900A0
.long 0x32161639
.long 0xBF8CC07F
.long 0x28000500
.long 0x28020701
.long 0x261A18BF
.long 0x201A1A84
.long 0x241A1A82
.long 0xD0C6003A, 0x0000150D
.long 0xD1000000, 0x00E90100
.long 0xD1000001, 0x00E90101
.long 0xD1000004, 0x00E90104
.long 0xD1000005, 0x00E90105
.long 0x6A1A1A0A
.long 0xD0C1003A, 0x0001090D
.long 0x8639830A
.long 0x80B93984
.long 0x8E398439
.long 0xD28F000E, 0x00020039
.long 0xD1000000, 0x00EA1D00
.long 0xD1000001, 0x00EA1F01
.long 0xD28F000E, 0x00020839
.long 0xD1000004, 0x00EA1D04
.long 0xD1000005, 0x00EA1F05
.long 0xBF800001
.long 0xD3CD8000, 0x04020900
.long 0x818A900A
.long 0x800B900B
.long 0xBF05800A
.long 0xBF84FFC8
.long 0xBF068126
.long 0xBF840004
.long 0xC00E0A00, 0x0000005C
.long 0xC0020C00, 0x0000007C
.long 0xBE980014
.long 0xBE990015
.long 0xBE9A00FF, 0x80000000
.long 0xBE9B00FF, 0x00020000
.long 0xBEA00016
.long 0xBEA10017
.long 0xBEA200FF, 0x80000000
.long 0xBEA300FF, 0x00020000
.long 0x923E0390
.long 0x963D1E3E
.long 0x923C1E3E
.long 0x8EBC053C
.long 0x80203C16
.long 0x82213D17
.long 0x963D1C3E
.long 0x923C1C3E
.long 0x8EBC083C
.long 0x80183C14
.long 0x82193D15
.long 0x963D1F04
.long 0x923C1F04
.long 0x8EBC053C
.long 0x80203C20
.long 0x82213D21
.long 0x963D1D04
.long 0x923C1D04
.long 0x8EBC083C
.long 0x80183C18
.long 0x82193D19
.long 0xBF068126
.long 0xBF850011
.long 0x963D0610
.long 0x923C0610
.long 0x80C08111
.long 0x92400640
.long 0x963F1E40
.long 0x923E1E40
.long 0x803C3E3C
.long 0x823D3F3D
.long 0x80C08112
.long 0x92400640
.long 0x963F1F40
.long 0x923E1F40
.long 0x803C3E3C
.long 0x823D3F3D
.long 0x8EBC823C
.long 0x80183C18
.long 0x82193D19
.long 0x20221886
.long 0x201C2281
.long 0xD285000E, 0x00021C90
.long 0x2624188F
.long 0xD1FE000E, 0x02021D12
.long 0xD285000F, 0x00003D0E
.long 0xD2850010, 0x0000390E
.long 0x26242281
.long 0xD2850012, 0x00022490
.long 0x261A18BF
.long 0x201A1A84
.long 0x241A1A82
.long 0xD1FE000D, 0x02021B12
.long 0x920502A0
.long 0x681A1A05
.long 0x92050390
.long 0x681C1C05
.long 0xBF8CC07F
.long 0xBF068126
.long 0xBF840006
.long 0xBEBC1C00
.long 0x813E84FF, 0x00000104
.long 0x803C3E3C
.long 0x823D803D
.long 0xBE801D3C
.long 0x863C109F
.long 0x803D0CC1
.long 0xBF093D02
.long 0x853C803C
.long 0xB53C0000
.long 0xBF850017
.long 0x863C118F
.long 0x803D0DC1
.long 0xBF093D03
.long 0x853C803C
.long 0xB53C0000
.long 0xBF850011
.long 0xD1FE0011, 0x020A1B10
.long 0xD3D84018, 0x18000100
.long 0xD3D84019, 0x18000101
.long 0xD3D8401A, 0x18000102
.long 0xD3D8401B, 0x18000103
.long 0xBF800001
.long 0xBF800000
.long 0xE07C1000, 0x80061811
.long 0xBF800000
.long 0xBF800000
.long 0xBF82001A
.long 0x7E3002FF, 0x80000000
.long 0xD0C9003C, 0x0000210D
.long 0xD0C90040, 0x0000230E
.long 0x86C0403C
.long 0xD1FE0011, 0x020A1B10
.long 0xD1000011, 0x01022318
.long 0xD3D84014, 0x18000100
.long 0xD3D84015, 0x18000101
.long 0xD3D84016, 0x18000102
.long 0xD3D84017, 0x18000103
.long 0xBF800001
.long 0xBF800000
.long 0xE07C1000, 0x80061411
.long 0xBF800000
.long 0xBF800000
.long 0xBF820000
.long 0xBEBC1C00
.long 0x813E84FF, 0x00000B64
.long 0x803C3E3C
.long 0x823D803D
.long 0xBE801D3C
.long 0xBEB40028
.long 0xBEB50029
.long 0xBEB700FF, 0x00020000
.long 0xBF128028
.long 0xBF840002
.long 0xBEB60080
.long 0xBF820001
.long 0xBEB60010
.long 0x92363684
.long 0x80058104
.long 0x9205052D
.long 0xBF068005
.long 0x85050510
.long 0xBEB8002A
.long 0xBEB9002B
.long 0xBEBB00FF, 0x00020000
.long 0xBF12802A
.long 0xBF840002
.long 0xBEBA0080
.long 0xBF820001
.long 0xBEBA0005
.long 0xB4AC0000
.long 0xBF85000E
.long 0x923A3A84
.long 0x920502A0
.long 0x68221805
.long 0x9205042D
.long 0x68222205
.long 0x24222282
.long 0xE0501000, 0x800E1211
.long 0x24221882
.long 0xBF8C0F70
.long 0xBF8A0000
.long 0xD81A0000, 0x00001211
.long 0xBF820011
.long 0xB4AC0004
.long 0xBF85000F
.long 0x923A3A82
.long 0x920502A0
.long 0x68221805
.long 0x9205042D
.long 0x68222205
.long 0x24222281
.long 0xE0901000, 0x800E1211
.long 0x24221882
.long 0xBF8C0F70
.long 0xBF8A0000
.long 0x7E241712
.long 0xD81A0000, 0x00001211
.long 0xBF820000
.long 0xB4300001
.long 0xBF850014
.long 0xB4300002
.long 0xBF850018
.long 0xB4300003
.long 0xBF85001C
.long 0xB4300004
.long 0xBF850020
.long 0xB4300005
.long 0xBF850024
.long 0xB4300006
.long 0xBF850028
.long 0xB4300007
.long 0xBF85002C
.long 0xB4300009
.long 0xBF850030
.long 0xBE8A1C00
.long 0x810584FF, 0x00000618
.long 0x800A050A
.long 0x820B800B
.long 0xBF820030
.long 0xBE8A1C00
.long 0x810584FF, 0x00000604
.long 0x800A050A
.long 0x820B800B
.long 0xBF82002A
.long 0xBE8A1C00
.long 0x810584FF, 0x00000610
.long 0x800A050A
.long 0x820B800B
.long 0xBF820024
.long 0xBE8A1C00
.long 0x810584FF, 0x0000063C
.long 0x800A050A
.long 0x820B800B
.long 0xBF82001E
.long 0xBE8A1C00
.long 0x810584FF, 0x00000728
.long 0x800A050A
.long 0x820B800B
.long 0xBF820018
.long 0xBE8A1C00
.long 0x810584FF, 0x00000754
.long 0x800A050A
.long 0x820B800B
.long 0xBF820012
.long 0xBE8A1C00
.long 0x810584FF, 0x00000760
.long 0x800A050A
.long 0x820B800B
.long 0xBF82000C
.long 0xBE8A1C00
.long 0x810584FF, 0x000007BC
.long 0x800A050A
.long 0x820B800B
.long 0xBF820006
.long 0xBE8A1C00
.long 0x810584FF, 0x00000858
.long 0x800A050A
.long 0x820B800B
.long 0xBF820000
.long 0xB4250000
.long 0xBF84009B
.long 0x863C109F
.long 0x803D0CC1
.long 0xBF093D02
.long 0x853C803C
.long 0xB53C0000
.long 0xBF850048
.long 0x863C118F
.long 0x803D0DC1
.long 0xBF093D03
.long 0x853C803C
.long 0xB53C0000
.long 0xBF850042
.long 0x923202A0
.long 0xD1350018, 0x0000650D
.long 0x24303082
.long 0xBF8CC07F
.long 0xBF8A0000
.long 0xD9FE0000, 0x1C000018
.long 0x24321A82
.long 0xE05C1000, 0x800D2019
.long 0xD1FE0011, 0x02061B10
.long 0xD3D84024, 0x18000100
.long 0xD3D84025, 0x18000101
.long 0xD3D84026, 0x18000102
.long 0xD3D84027, 0x18000103
.long 0xBF800001
.long 0xBF800000
.long 0x0A484824
.long 0x0A4A4A24
.long 0x0A4C4C24
.long 0x0A4E4E24
.long 0xBF8C0000
.long 0xD0CC0028, 0x00010036
.long 0xD1000020, 0x00A240F2
.long 0xD1000021, 0x00A242F2
.long 0xD3B14024, 0x18024920
.long 0xD0CC0028, 0x00010036
.long 0xD1000022, 0x00A244F2
.long 0xD1000023, 0x00A246F2
.long 0xD3B14026, 0x18024D22
.long 0xD3B24012, 0x1802491C
.long 0xD3B24014, 0x18024D1E
.long 0xBE8E1E0A
.long 0x7E480312
.long 0x7E4A0313
.long 0x7E4C0314
.long 0x7E4E0315
.long 0x7E481524
.long 0x7E4A1525
.long 0xD2A00024, 0x00024B24
.long 0x7E4C1526
.long 0x7E4E1527
.long 0xD2A00025, 0x00024F26
.long 0xE0741000, 0x80062411
.long 0xBF800000
.long 0xBF800000
.long 0xBF820207
.long 0x7E3202FF, 0x80000000
.long 0xD0C9003C, 0x0000210D
.long 0xD0C90040, 0x0000230E
.long 0x86C0403C
.long 0x923C02A0
.long 0xD1350017, 0x0000790D
.long 0x242E2E82
.long 0xD1000017, 0x01022F19
.long 0xBF8CC07F
.long 0xBF8A0000
.long 0xD9FE0000, 0x1C000017
.long 0x24301A82
.long 0xE05C1000, 0x800D2018
.long 0xD1FE0011, 0x02061B10
.long 0xD1000011, 0x01022319
.long 0xD3D84024, 0x18000100
.long 0xD3D84025, 0x18000101
.long 0xD3D84026, 0x18000102
.long 0xD3D84027, 0x18000103
.long 0xBF800001
.long 0xBF800000
.long 0x0A484824
.long 0x0A4A4A24
.long 0x0A4C4C24
.long 0x0A4E4E24
.long 0xBF8C0000
.long 0xD0CC0028, 0x00010036
.long 0xD1000020, 0x00A240F2
.long 0xD1000021, 0x00A242F2
.long 0xD3B14024, 0x18024920
.long 0xD0CC0028, 0x00010036
.long 0xD1000022, 0x00A244F2
.long 0xD1000023, 0x00A246F2
.long 0xD3B14026, 0x18024D22
.long 0xD3B24012, 0x1802491C
.long 0xD3B24014, 0x18024D1E
.long 0xBE8E1E0A
.long 0x7E480312
.long 0x7E4A0313
.long 0x7E4C0314
.long 0x7E4E0315
.long 0x7E481524
.long 0x7E4A1525
.long 0xD2A00024, 0x00024B24
.long 0x7E4C1526
.long 0x7E4E1527
.long 0xD2A00025, 0x00024F26
.long 0xE0741000, 0x80062411
.long 0xBF800000
.long 0xBF800000
.long 0xBF8201BA
.long 0x863C109F
.long 0x803D0CC1
.long 0xBF093D02
.long 0x853C803C
.long 0xB53C0000
.long 0xBF850054
.long 0x863C118F
.long 0x803D0DC1
.long 0xBF093D03
.long 0x853C803C
.long 0xB53C0000
.long 0xBF85004E
.long 0xD1FE0017, 0x02061B0F
.long 0xE0541000, 0x80081A17
.long 0x923202A0
.long 0xD1350018, 0x0000650D
.long 0x24303082
.long 0xBF8CC07F
.long 0xBF8A0000
.long 0xD9FE0000, 0x1C000018
.long 0x24321A82
.long 0xE05C1000, 0x800D2019
.long 0xD1FE0011, 0x02061B10
.long 0xD3D84024, 0x18000100
.long 0xD3D84025, 0x18000101
.long 0xD3D84026, 0x18000102
.long 0xD3D84027, 0x18000103
.long 0xBF800001
.long 0xBF800000
.long 0x0A484824
.long 0x0A4A4A24
.long 0x0A4C4C24
.long 0x0A4E4E24
.long 0xBF8C0000
.long 0xD0CC0028, 0x00010036
.long 0xD1000020, 0x00A240F2
.long 0xD1000021, 0x00A242F2
.long 0xD3B14024, 0x18024920
.long 0xD0CC0028, 0x00010036
.long 0xD1000022, 0x00A244F2
.long 0xD1000023, 0x00A246F2
.long 0xD3B14026, 0x18024D22
.long 0xD3A00024, 0x14923425
.long 0xD3A01025, 0x14963425
.long 0xD3A00026, 0x149A3625
.long 0xD3A01027, 0x149E3625
.long 0xD3B24012, 0x1802491C
.long 0xD3B24014, 0x18024D1E
.long 0xBE8E1E0A
.long 0x7E480312
.long 0x7E4A0313
.long 0x7E4C0314
.long 0x7E4E0315
.long 0x7E481524
.long 0x7E4A1525
.long 0xD2A00024, 0x00024B24
.long 0x7E4C1526
.long 0x7E4E1527
.long 0xD2A00025, 0x00024F26
.long 0xE0741000, 0x80062411
.long 0xBF800000
.long 0xBF800000
.long 0xBF820160
.long 0x7E3202FF, 0x80000000
.long 0xD0C9003C, 0x0000210D
.long 0xD0C90040, 0x0000230E
.long 0x86C0403C
.long 0xD1FE0011, 0x02061B0F
.long 0xD1000011, 0x01022319
.long 0xE0541000, 0x80081A11
.long 0x923C02A0
.long 0xD1350017, 0x0000790D
.long 0x242E2E82
.long 0xD1000017, 0x01022F19
.long 0xBF8CC07F
.long 0xBF8A0000
.long 0xD9FE0000, 0x1C000017
.long 0x24301A82
.long 0xE05C1000, 0x800D2018
.long 0xD1FE0011, 0x02061B10
.long 0xD1000011, 0x01022319
.long 0xD3D84024, 0x18000100
.long 0xD3D84025, 0x18000101
.long 0xD3D84026, 0x18000102
.long 0xD3D84027, 0x18000103
.long 0xBF800001
.long 0xBF800000
.long 0x0A484824
.long 0x0A4A4A24
.long 0x0A4C4C24
.long 0x0A4E4E24
.long 0xBF8C0000
.long 0xD0CC0028, 0x00010036
.long 0xD1000020, 0x00A240F2
.long 0xD1000021, 0x00A242F2
.long 0xD3B14024, 0x18024920
.long 0xD0CC0028, 0x00010036
.long 0xD1000022, 0x00A244F2
.long 0xD1000023, 0x00A246F2
.long 0xD3B14026, 0x18024D22
.long 0xD3A00024, 0x14923425
.long 0xD3A01025, 0x14963425
.long 0xD3A00026, 0x149A3625
.long 0xD3A01027, 0x149E3625
.long 0xD3B24012, 0x1802491C
.long 0xD3B24014, 0x18024D1E
.long 0xBE8E1E0A
.long 0x7E480312
.long 0x7E4A0313
.long 0x7E4C0314
.long 0x7E4E0315
.long 0x7E481524
.long 0x7E4A1525
.long 0xD2A00024, 0x00024B24
.long 0x7E4C1526
.long 0x7E4E1527
.long 0xD2A00025, 0x00024F26
.long 0xE0741000, 0x80062411
.long 0xBF800000
.long 0xBF800000
.long 0xBF820105
.long 0xBE801D0E
.long 0x262424FF, 0x7FFFFFFF
.long 0x262626FF, 0x7FFFFFFF
.long 0x262828FF, 0x7FFFFFFF
.long 0x262A2AFF, 0x7FFFFFFF
.long 0xBE801D0E
.long 0xD044006A, 0x00005D12
.long 0x1424242F
.long 0x00242480
.long 0xD044006A, 0x00005D13
.long 0x1426262F
.long 0x00262680
.long 0xD044006A, 0x00005D14
.long 0x1428282F
.long 0x00282880
.long 0xD044006A, 0x00005D15
.long 0x142A2A2F
.long 0x002A2A80
.long 0xBE801D0E
.long 0x0A2C24FF, 0x3D372713
.long 0xD1CB0016, 0x03CA2D12
.long 0x0A2C2D12
.long 0x0A2C2CFF, 0x40135761
.long 0x7E2C4116
.long 0xBF800000
.long 0x022C2CF2
.long 0x7E2C4516
.long 0xBF800000
.long 0xD1CB0016, 0x03D22CF5
.long 0x0A2C2D12
.long 0x0A242CF0
.long 0x0A2C26FF, 0x3D372713
.long 0xD1CB0016, 0x03CA2D13
.long 0x0A2C2D13
.long 0x0A2C2CFF, 0x40135761
.long 0x7E2C4116
.long 0xBF800000
.long 0x022C2CF2
.long 0x7E2C4516
.long 0xBF800000
.long 0xD1CB0016, 0x03D22CF5
.long 0x0A2C2D13
.long 0x0A262CF0
.long 0x0A2C28FF, 0x3D372713
.long 0xD1CB0016, 0x03CA2D14
.long 0x0A2C2D14
.long 0x0A2C2CFF, 0x40135761
.long 0x7E2C4116
.long 0xBF800000
.long 0x022C2CF2
.long 0x7E2C4516
.long 0xBF800000
.long 0xD1CB0016, 0x03D22CF5
.long 0x0A2C2D14
.long 0x0A282CF0
.long 0x0A2C2AFF, 0x3D372713
.long 0xD1CB0016, 0x03CA2D15
.long 0x0A2C2D15
.long 0x0A2C2CFF, 0x40135761
.long 0x7E2C4116
.long 0xBF800000
.long 0x022C2CF2
.long 0x7E2C4516
.long 0xBF800000
.long 0xD1CB0016, 0x03D22CF5
.long 0x0A2C2D15
.long 0x0A2A2CF0
.long 0xBE801D0E
.long 0x0A2C242E
.long 0xD046006A, 0x00010112
.long 0x00242516
.long 0x0A2C262E
.long 0xD046006A, 0x00010113
.long 0x00262716
.long 0x0A2C282E
.long 0xD046006A, 0x00010114
.long 0x00282916
.long 0x0A2C2A2E
.long 0xD046006A, 0x00010115
.long 0x002A2B16
.long 0xBE801D0E
.long 0xD10B0012, 0x00010112
.long 0xD10B0013, 0x00010113
.long 0xD10B0014, 0x00010114
.long 0xD10B0015, 0x00010115
.long 0xBE801D0E
.long 0x0A2424FF, 0xBFB8AA3B
.long 0x7E244112
.long 0xBF800000
.long 0x022424F2
.long 0x7E244512
.long 0xBF800000
.long 0x0A2626FF, 0xBFB8AA3B
.long 0x7E264113
.long 0xBF800000
.long 0x022626F2
.long 0x7E264513
.long 0xBF800000
.long 0x0A2828FF, 0xBFB8AA3B
.long 0x7E284114
.long 0xBF800000
.long 0x022828F2
.long 0x7E284514
.long 0xBF800000
.long 0x0A2A2AFF, 0xBFB8AA3B
.long 0x7E2A4115
.long 0xBF800000
.long 0x022A2AF2
.long 0x7E2A4515
.long 0xBF800000
.long 0xBE801D0E
.long 0x0A24242E
.long 0x0A2424FF, 0x4038AA3B
.long 0x7E244112
.long 0xBF800000
.long 0x022424F2
.long 0x7E244512
.long 0xBF800000
.long 0xD1CB0012, 0x03CA24F5
.long 0x0A24242F
.long 0x0A26262E
.long 0x0A2626FF, 0x4038AA3B
.long 0x7E264113
.long 0xBF800000
.long 0x022626F2
.long 0x7E264513
.long 0xBF800000
.long 0xD1CB0013, 0x03CA26F5
.long 0x0A26262F
.long 0x0A28282E
.long 0x0A2828FF, 0x4038AA3B
.long 0x7E284114
.long 0xBF800000
.long 0x022828F2
.long 0x7E284514
.long 0xBF800000
.long 0xD1CB0014, 0x03CA28F5
.long 0x0A28282F
.long 0x0A2A2A2E
.long 0x0A2A2AFF, 0x4038AA3B
.long 0x7E2A4115
.long 0xBF800000
.long 0x022A2AF2
.long 0x7E2A4515
.long 0xBF800000
.long 0xD1CB0015, 0x03CA2AF5
.long 0x0A2A2A2F
.long 0xBE801D0E
.long 0x0A2C24FF, 0x3D372713
.long 0xD1CB0016, 0x03CA2D12
.long 0x0A2C2D12
.long 0x0A2C2CFF, 0x40135761
.long 0x7E2C4116
.long 0xBF800000
.long 0x022C2CF2
.long 0x7E2C4516
.long 0xBF800000
.long 0xD1CB0016, 0x03D22CF5
.long 0x0A2C2D12
.long 0x0A2C2CF0
.long 0x0A242C2E
.long 0x0A2C26FF, 0x3D372713
.long 0xD1CB0016, 0x03CA2D13
.long 0x0A2C2D13
.long 0x0A2C2CFF, 0x40135761
.long 0x7E2C4116
.long 0xBF800000
.long 0x022C2CF2
.long 0x7E2C4516
.long 0xBF800000
.long 0xD1CB0016, 0x03D22CF5
.long 0x0A2C2D13
.long 0x0A2C2CF0
.long 0x0A262C2E
.long 0x0A2C28FF, 0x3D372713
.long 0xD1CB0016, 0x03CA2D14
.long 0x0A2C2D14
.long 0x0A2C2CFF, 0x40135761
.long 0x7E2C4116
.long 0xBF800000
.long 0x022C2CF2
.long 0x7E2C4516
.long 0xBF800000
.long 0xD1CB0016, 0x03D22CF5
.long 0x0A2C2D14
.long 0x0A2C2CF0
.long 0x0A282C2E
.long 0x0A2C2AFF, 0x3D372713
.long 0xD1CB0016, 0x03CA2D15
.long 0x0A2C2D15
.long 0x0A2C2CFF, 0x40135761
.long 0x7E2C4116
.long 0xBF800000
.long 0x022C2CF2
.long 0x7E2C4516
.long 0xBF800000
.long 0xD1CB0016, 0x03D22CF5
.long 0x0A2C2D15
.long 0x0A2C2CF0
.long 0x0A2A2C2E
.long 0xBE801D0E
.long 0xBF810000
