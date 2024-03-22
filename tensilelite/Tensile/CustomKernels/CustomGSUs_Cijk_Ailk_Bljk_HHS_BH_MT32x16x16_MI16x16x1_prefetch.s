
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
.text
.protected CustomGSUs_Cijk_Ailk_Bljk_HHS_BH_MT32x16x16_MI16x16x1_prefetch
.globl CustomGSUs_Cijk_Ailk_Bljk_HHS_BH_MT32x16x16_MI16x16x1_prefetch
.p2align 8
.type CustomGSUs_Cijk_Ailk_Bljk_HHS_BH_MT32x16x16_MI16x16x1_prefetch,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel CustomGSUs_Cijk_Ailk_Bljk_HHS_BH_MT32x16x16_MI16x16x1_prefetch
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_accum_offset 48 // accvgpr offset
  .amdhsa_next_free_vgpr 52 // vgprs
  .amdhsa_next_free_sgpr 62 // sgprs
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
/* Num VGPR   =44 */
/* Num AccVGPR=4 */
/* Num SGPR   =62 */

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
  - .name: CustomGSUs_Cijk_Ailk_Bljk_HHS_BH_MT32x16x16_MI16x16x1_prefetch
    .symbol: 'CustomGSUs_Cijk_Ailk_Bljk_HHS_BH_MT32x16x16_MI16x16x1_prefetch.kd'
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
    .group_segment_fixed_size:   1824
    .kernarg_segment_align:      8
    .kernarg_segment_size:       96
    .max_flat_workgroup_size:    128
    .private_segment_fixed_size: 0
    .sgpr_count:                 62
    .sgpr_spill_count:           0
    .vgpr_count:                 44
    .vgpr_spill_count:           0
    .wavefront_size:             64
...
.end_amdgpu_metadata
CustomGSUs_Cijk_Ailk_Bljk_HHS_BH_MT32x16x16_MI16x16x1_prefetch:

.long 0xC0120400, 0x00000000
.long 0xC00A0800, 0x00000040
.long 0xC0060900, 0x00000050
.long 0xC0020980, 0x00000058
.long 0xBEFC00FF, 0x00000720
.long 0x7E180300
.long 0xBEA81C00
.long 0xBEAA00C1
.long 0xBEAB00FF, 0x00020000
.long 0xD1FD002E, 0x02010F0C
.long 0xE0501000, 0x800A2F2E
.long 0xD1FD002E, 0x04B90F0C
.long 0xE0501000, 0x800A2F2E
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
.long 0xBE981C00
.long 0xBE9A00C1
.long 0xBE9B00FF, 0x00020000
.long 0xD1FD002E, 0x02010F0C
.long 0xE0501000, 0x80062F2E
.long 0xD1FD002E, 0x04B90F0C
.long 0xE0501000, 0x80062F2E
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
.long 0xBE980014
.long 0xBE990015
.long 0xBE9A00FF, 0x80000000
.long 0xBE9B00FF, 0x00020000
.long 0xBEA00016
.long 0xBEA10017
.long 0xBEA200FF, 0x80000000
.long 0xBEA300FF, 0x00020000
.long 0x922A0390
.long 0x96291E2A
.long 0x92281E2A
.long 0x8EA80528
.long 0x80202816
.long 0x82212917
.long 0x96291C2A
.long 0x92281C2A
.long 0x8EA80828
.long 0x80182814
.long 0x82192915
.long 0x96291F04
.long 0x92281F04
.long 0x8EA80528
.long 0x80202820
.long 0x82212921
.long 0x96291D04
.long 0x92281D04
.long 0x8EA80828
.long 0x80182818
.long 0x82192919
.long 0xBF068126
.long 0xBF850011
.long 0x96290610
.long 0x92280610
.long 0x80AB8111
.long 0x922B062B
.long 0x962A1E2B
.long 0x92271E2B
.long 0x80282728
.long 0x82292A29
.long 0x80AB8112
.long 0x922B062B
.long 0x962A1F2B
.long 0x92271F2B
.long 0x80282728
.long 0x82292A29
.long 0x8EA88228
.long 0x80182818
.long 0x82192919
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
.long 0xBF068126
.long 0xBF840006
.long 0xBEA81C00
.long 0x812A84FF, 0x00000104
.long 0x80282A28
.long 0x82298029
.long 0xBE801D28
.long 0x8628109F
.long 0x80290CC1
.long 0xBF092902
.long 0x85288028
.long 0xB5280000
.long 0xBF850017
.long 0x8628118F
.long 0x80290DC1
.long 0xBF092903
.long 0x85288028
.long 0xB5280000
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
.long 0xD0C90028, 0x0000210D
.long 0xD0C9002C, 0x0000230E
.long 0x86AC2C28
.long 0xD1FE0011, 0x020A1B10
.long 0xD1000011, 0x00B22318
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
.long 0xBEA81C00
.long 0x812A84FF, 0x00000300
.long 0x80282A28
.long 0x82298029
.long 0xBE801D28
.long 0xB4250000
.long 0xBF84004F
.long 0x8628109F
.long 0x80290CC1
.long 0xBF092902
.long 0x85288028
.long 0xB5280000
.long 0xBF850023
.long 0x8628118F
.long 0x80290DC1
.long 0xBF092903
.long 0x85288028
.long 0xB5280000
.long 0xBF85001D
.long 0xD1FE0011, 0x02061B10
.long 0xD3D84018, 0x18000100
.long 0xD3D84019, 0x18000101
.long 0xD3D8401A, 0x18000102
.long 0xD3D8401B, 0x18000103
.long 0xBF800001
.long 0xBF800000
.long 0x0A303024
.long 0x0A323224
.long 0x0A343424
.long 0x0A363624
.long 0x7E301518
.long 0x7E321519
.long 0xD2A00018, 0x00023318
.long 0x7E34151A
.long 0x7E36151B
.long 0xD2A00019, 0x0002371A
.long 0xE0741000, 0x80061811
.long 0xBF800000
.long 0xBF800000
.long 0xBF820091
.long 0x7E3002FF, 0x80000000
.long 0xD0C90028, 0x0000210D
.long 0xD0C9002C, 0x0000230E
.long 0x86AC2C28
.long 0xD1FE0011, 0x02061B10
.long 0xD1000011, 0x00B22318
.long 0xD3D84014, 0x18000100
.long 0xD3D84015, 0x18000101
.long 0xD3D84016, 0x18000102
.long 0xD3D84017, 0x18000103
.long 0xBF800001
.long 0xBF800000
.long 0x0A282824
.long 0x0A2A2A24
.long 0x0A2C2C24
.long 0x0A2E2E24
.long 0x7E281514
.long 0x7E2A1515
.long 0xD2A00014, 0x00022B14
.long 0x7E2C1516
.long 0x7E2E1517
.long 0xD2A00015, 0x00022F16
.long 0xE0741000, 0x80061411
.long 0xBF800000
.long 0xBF800000
.long 0xBF82006B
.long 0x8628109F
.long 0x80290CC1
.long 0xBF092902
.long 0x85288028
.long 0xB5280000
.long 0xBF850030
.long 0x8628118F
.long 0x80290DC1
.long 0xBF092903
.long 0x85288028
.long 0xB5280000
.long 0xBF85002A
.long 0xD1FE0014, 0x02061B0F
.long 0xE0541000, 0x80081614
.long 0xD1FE0011, 0x02061B10
.long 0xD3D84018, 0x18000100
.long 0xD3D84019, 0x18000101
.long 0xD3D8401A, 0x18000102
.long 0xD3D8401B, 0x18000103
.long 0xBF800001
.long 0xBF800000
.long 0x0A303024
.long 0x0A323224
.long 0x0A343424
.long 0x0A363624
.long 0xBF8C0F70
.long 0xD3A00018, 0x14622C25
.long 0xD3A01019, 0x14662C25
.long 0xD3A0001A, 0x146A2E25
.long 0xD3A0101B, 0x146E2E25
.long 0x7E301518
.long 0x7E321519
.long 0xD2A00018, 0x00023318
.long 0x7E34151A
.long 0x7E36151B
.long 0xD2A00019, 0x0002371A
.long 0xE0741000, 0x80061811
.long 0xBF800000
.long 0xBF800000
.long 0xBF820035
.long 0x7E2C02FF, 0x80000000
.long 0xD0C90028, 0x0000210D
.long 0xD0C9002C, 0x0000230E
.long 0x86AC2C28
.long 0xD1FE0011, 0x02061B0F
.long 0xD1000011, 0x00B22316
.long 0xE0541000, 0x80081411
.long 0xD1FE0011, 0x02061B10
.long 0xD1000011, 0x00B22316
.long 0xD3D84018, 0x18000100
.long 0xD3D84019, 0x18000101
.long 0xD3D8401A, 0x18000102
.long 0xD3D8401B, 0x18000103
.long 0xBF800001
.long 0xBF800000
.long 0x0A303024
.long 0x0A323224
.long 0x0A343424
.long 0x0A363624
.long 0xBF8C0F70
.long 0xD3A00018, 0x14622825
.long 0xD3A01019, 0x14662825
.long 0xD3A0001A, 0x146A2A25
.long 0xD3A0101B, 0x146E2A25
.long 0x7E301518
.long 0x7E321519
.long 0xD2A00018, 0x00023318
.long 0x7E34151A
.long 0x7E36151B
.long 0xD2A00019, 0x0002371A
.long 0xE0741000, 0x80061811
.long 0xBF800000
.long 0xBF800000
.long 0xBF820000
.long 0xBF810000
