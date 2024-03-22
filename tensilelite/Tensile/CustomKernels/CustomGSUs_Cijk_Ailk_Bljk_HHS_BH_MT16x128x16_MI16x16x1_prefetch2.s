
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
.text
.protected CustomGSUs_Cijk_Ailk_Bljk_HHS_BH_MT16x128x16_MI16x16x1_prefetch2
.globl CustomGSUs_Cijk_Ailk_Bljk_HHS_BH_MT16x128x16_MI16x16x1_prefetch2
.p2align 8
.type CustomGSUs_Cijk_Ailk_Bljk_HHS_BH_MT16x128x16_MI16x16x1_prefetch2,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel CustomGSUs_Cijk_Ailk_Bljk_HHS_BH_MT16x128x16_MI16x16x1_prefetch2
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_accum_offset 40 // accvgpr offset
  .amdhsa_next_free_vgpr 48 // vgprs
  .amdhsa_next_free_sgpr 62 // sgprs
  .amdhsa_group_segment_fixed_size 5120 // lds bytes
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel
.text
/* Num VGPR   =40 */
/* Num AccVGPR=8 */
/* Num SGPR   =62 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 4 x 2 */
/* SubGroup= 4 x 64 */
/* VectorWidthA=1 */
/* VectorWidthB=1 */
/* GlobalReadVectorWidthA=1, GlobalReadVectorWidthB=4 */
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
  - .name: CustomGSUs_Cijk_Ailk_Bljk_HHS_BH_MT16x128x16_MI16x16x1_prefetch2
    .symbol: 'CustomGSUs_Cijk_Ailk_Bljk_HHS_BH_MT16x128x16_MI16x16x1_prefetch2.kd'
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
    .group_segment_fixed_size:   5120
    .kernarg_segment_align:      8
    .kernarg_segment_size:       96
    .max_flat_workgroup_size:    256
    .private_segment_fixed_size: 0
    .sgpr_count:                 62
    .sgpr_spill_count:           0
    .vgpr_count:                 40
    .vgpr_spill_count:           0
    .wavefront_size:             64
...
.end_amdgpu_metadata
CustomGSUs_Cijk_Ailk_Bljk_HHS_BH_MT16x128x16_MI16x16x1_prefetch2:

.long 0xC0120400, 0x00000000
.long 0xC00A0800, 0x00000040
.long 0xC0060900, 0x00000050
.long 0xC0020980, 0x00000058
.long 0xBEFC00FF, 0x00001400
.long 0x7E1C0300
.long 0xBEA81C00
.long 0xBEAA00C1
.long 0xBEAB00FF, 0x00020000
.long 0xD1FD0026, 0x02010F0E
.long 0xE0501000, 0x800A2726
.long 0xD1FD0026, 0x04990F0E
.long 0xE0501000, 0x800A2726
.long 0x26201CBF
.long 0x261E208F
.long 0x26201CBF
.long 0x20202084
.long 0x24202086
.long 0x681E1F10
.long 0x26221CBF
.long 0x2620228F
.long 0x24202084
.long 0x26221CBF
.long 0x20222284
.long 0x24222282
.long 0x68202111
.long 0x20221C86
.long 0x26222283
.long 0x24222288
.long 0x68202111
.long 0x20221C86
.long 0x20222282
.long 0xBEBA00FF, 0x00000100
.long 0xD2850011, 0x0002223A
.long 0xD1FE000C, 0x02061F11
.long 0x20241887
.long 0x24242485
.long 0x68181912
.long 0x201E1C86
.long 0x201E1E82
.long 0xBEBA0090
.long 0xD285000F, 0x00021E3A
.long 0xD1FE000D, 0x0206210F
.long 0x20221A87
.long 0x24222283
.long 0x681A1B11
.long 0x321A1AFF, 0x00000300
.long 0x20201C84
.long 0x261E1C8F
.long 0x7E220310
.long 0x20241C82
.long 0x26261C83
.long 0x24262682
.long 0x7E2C0313
.long 0x10102290
.long 0xD1FE0008, 0x0206110F
.long 0x20281087
.long 0x24282885
.long 0x68101114
.long 0x10122490
.long 0xD1FE0009, 0x02061316
.long 0x20281287
.long 0x24282883
.long 0x68121314
.long 0x321212FF, 0x00000300
.long 0xBF8CC07F
.long 0x8609FF26, 0x0000FF00
.long 0x8F098809
.long 0x8627FF26, 0xFFFF0000
.long 0x8F279027
.long 0x8626FF26, 0x000000FF
.long 0x7E320290
.long 0x7E300210
.long 0x7E2E0D19
.long 0x7E2E4717
.long 0x7E340D18
.long 0x0A2E3517
.long 0x7E2E0F17
.long 0x10343317
.long 0x6A343518
.long 0xD0CD006A, 0x0001011A
.long 0xD11C6A17, 0x01A90117
.long 0x7E3202FF, 0x00000080
.long 0x7E300211
.long 0x7E180517
.long 0x7E2E0D19
.long 0x7E2E4717
.long 0x7E340D18
.long 0x0A2E3517
.long 0x7E2E0F17
.long 0x10343317
.long 0x6A343518
.long 0xD0CD006A, 0x0001011A
.long 0xD11C6A17, 0x01A90117
.long 0xBF800000
.long 0x7E1A0517
.long 0x80988218
.long 0x82998019
.long 0x809A881A
.long 0x829B801B
.long 0xD042006A, 0x00010024
.long 0xBF860001
.long 0xBE930080
.long 0xBF068126
.long 0xBF850012
.long 0x7E280C26
.long 0x7E284714
.long 0x7E2A0C03
.long 0x0A282B14
.long 0x7E280F14
.long 0xD1080015, 0x00004D14
.long 0x6A2A2A03
.long 0xD0DA007E, 0x00004D15
.long 0x68282881
.long 0x7E2A0280
.long 0xBEFE01C1
.long 0x7E060514
.long 0x7E0C0515
.long 0xBE850081
.long 0xBE880082
.long 0xBF820003
.long 0xBE860180
.long 0xBE850081
.long 0xBE880081
.long 0xBF0B8109
.long 0xBF850035
.long 0x7E280C09
.long 0x7E284714
.long 0x7E2A0C03
.long 0x0A282B14
.long 0x7E280F14
.long 0xD1080015, 0x00001314
.long 0x6A2A2A03
.long 0xD0DA007E, 0x00001315
.long 0x68282881
.long 0xBEFE01C1
.long 0x7E780514
.long 0x923D093C
.long 0x80BD3D03
.long 0x923D0C3D
.long 0x803D023D
.long 0x7E280C09
.long 0x7E284714
.long 0x7E2A0C0D
.long 0x0A282B14
.long 0x7E280F14
.long 0xD1080015, 0x00001314
.long 0x6A2A2A0D
.long 0xD0DA007E, 0x00001315
.long 0x68282881
.long 0xBEFE01C1
.long 0x7E740514
.long 0x923B3A09
.long 0x80BB3B0D
.long 0xBF06803B
.long 0xBEBB0209
.long 0xBF093A3C
.long 0x853A093B
.long 0x7E280C3A
.long 0x7E284714
.long 0x7E2A0C3D
.long 0x0A282B14
.long 0x7E280F14
.long 0xD1080015, 0x00007514
.long 0x6A2A2A3D
.long 0xD0DA007E, 0x00007515
.long 0x68282881
.long 0x7E2A0280
.long 0xBEFE01C1
.long 0x7E040514
.long 0x7E060515
.long 0x923C093C
.long 0x80033C03
.long 0xD2850017, 0x00022020
.long 0x32142F0F
.long 0x68141481
.long 0x24141481
.long 0xD285000F, 0x00022422
.long 0x32161F13
.long 0x68161684
.long 0x24161681
.long 0x9239C022
.long 0x8E398139
.long 0x963D9002
.long 0x923C9002
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
.long 0x800E820E
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
.long 0x963DFF03, 0x00000080
.long 0x923CFF03, 0x00000080
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
.long 0x80308830
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
.long 0x923AA026
.long 0x9237203A
.long 0x923AA026
.long 0xBEB8003A
.long 0xD3D94000, 0x18000080
.long 0xD3D94001, 0x18000080
.long 0xD3D94002, 0x18000080
.long 0xD3D94003, 0x18000080
.long 0xD3D94004, 0x18000080
.long 0xD3D94005, 0x18000080
.long 0xD3D94006, 0x18000080
.long 0xD3D94007, 0x18000080
.long 0x8F0A8413
.long 0xBF068126
.long 0xBF850012
.long 0x7E1E0C26
.long 0x7E1E470F
.long 0x7E200C0A
.long 0x0A1E210F
.long 0x7E1E0F0F
.long 0xD1080010, 0x00004D0F
.long 0x6A20200A
.long 0xD0DA007E, 0x00004D10
.long 0x681E1E81
.long 0x7E200280
.long 0xBEFE01C1
.long 0x7E14050F
.long 0x7E0E0510
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
.long 0xBF85003C
.long 0xE0901000, 0x800A000A
.long 0xE0541000, 0x800B040B
.long 0xE0541000, 0x390B060B
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
.long 0xD83E0000, 0x00000008
.long 0xD89A0000, 0x00000409
.long 0xD89A0880, 0x00000609
.long 0xBF8CC07F
.long 0xBF8C0000
.long 0xBF8A0000
.long 0xD8780000, 0x0000000C
.long 0xD8B60020, 0x0200000C
.long 0xD8780040, 0x0100000C
.long 0xD8B60060, 0x0300000C
.long 0xD8EC0000, 0x0400000D
.long 0xD8EC0880, 0x0600000D
.long 0xBF8CC07F
.long 0x28000500
.long 0x28020701
.long 0xBF800000
.long 0xBF800000
.long 0xD3CD8000, 0x04020900
.long 0xD3CD8004, 0x04120D00
.long 0x808A810A
.long 0xBF00800A
.long 0xBF84FFC4
.long 0xBE981C00
.long 0xBE9A00C1
.long 0xBE9B00FF, 0x00020000
.long 0xD1FD0026, 0x02010F0E
.long 0xE0501000, 0x80062726
.long 0xD1FD0026, 0x04990F0E
.long 0xE0501000, 0x80062726
.long 0x860A138F
.long 0xBF070706
.long 0xBE8A0280
.long 0xBF06800A
.long 0xBE8B0080
.long 0xBF850082
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
.long 0xE0901000, 0x800A000A
.long 0xE0901000, 0x800B040B
.long 0xE0941002, 0x800B0F0B
.long 0xBF8C0F70
.long 0x28081F04
.long 0xE0901004, 0x800B050B
.long 0xE0941006, 0x800B0F0B
.long 0xBF8C0F70
.long 0x280A1F05
.long 0xE0901000, 0x390B060B
.long 0xE0941002, 0x390B0F0B
.long 0xBF8C0F70
.long 0x280C1F06
.long 0xE0901004, 0x390B070B
.long 0xE0941006, 0x390B0F0B
.long 0xBF8C0F70
.long 0x280E1F07
.long 0xBF8C0F70
.long 0xBF8C0000
.long 0xBF8A0000
.long 0xD83E0000, 0x00000008
.long 0xD89A0000, 0x00000409
.long 0xD89A0880, 0x00000609
.long 0xBF8CC07F
.long 0xBF8C0000
.long 0xBF8A0000
.long 0xD8780000, 0x0000000C
.long 0xD8B60020, 0x0200000C
.long 0xD8780040, 0x0100000C
.long 0xD8B60060, 0x0300000C
.long 0xD8EC0000, 0x0400000D
.long 0xD8EC0880, 0x0600000D
.long 0xBEBA00FF, 0x00000280
.long 0x3218183A
.long 0xBEBA00A0
.long 0x321A1A3A
.long 0xBF8CC07F
.long 0x28000500
.long 0x28020701
.long 0x261E1CBF
.long 0x201E1E84
.long 0x241E1E82
.long 0xD0C6003A, 0x0000150F
.long 0xD1000000, 0x00E90100
.long 0xD1000001, 0x00E90101
.long 0xD1000004, 0x00E90104
.long 0xD1000006, 0x00E90106
.long 0xD1000005, 0x00E90105
.long 0xD1000007, 0x00E90107
.long 0x6A1E1E0A
.long 0xD0C1003A, 0x0001090F
.long 0x863C830A
.long 0x80BC3C84
.long 0x8E3C843C
.long 0xD28F0010, 0x0002003C
.long 0xD1000000, 0x00EA2100
.long 0xD1000001, 0x00EA2301
.long 0xD28F0010, 0x0002083C
.long 0xD1000004, 0x00EA2104
.long 0xD1000005, 0x00EA2305
.long 0xD28F0010, 0x00020C3C
.long 0xD1000006, 0x00EA2106
.long 0xD1000007, 0x00EA2307
.long 0xBF800001
.long 0xD3CD8000, 0x04020900
.long 0xD3CD8004, 0x04120D00
.long 0x818A900A
.long 0x800B900B
.long 0xBF05800A
.long 0xBF84FFBA
.long 0xBE980014
.long 0xBE990015
.long 0xBE9A00FF, 0x80000000
.long 0xBE9B00FF, 0x00020000
.long 0xBEA00016
.long 0xBEA10017
.long 0xBEA200FF, 0x80000000
.long 0xBEA300FF, 0x00020000
.long 0x922A03FF, 0x00000080
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
.long 0x20261C86
.long 0x20202680
.long 0xD2850010, 0x00022090
.long 0x26281C8F
.long 0xD1FE0010, 0x02022114
.long 0xD2850011, 0x00003D10
.long 0xD2850012, 0x00003910
.long 0x26282680
.long 0xD2850014, 0x00022890
.long 0x261E1CBF
.long 0x201E1E84
.long 0x241E1E82
.long 0xD1FE000F, 0x02021F14
.long 0x92050290
.long 0x681E1E05
.long 0x920503FF, 0x00000080
.long 0x68202005
.long 0xBF068126
.long 0xBF840006
.long 0xBEA81C00
.long 0x812A84FF, 0x000001B4
.long 0x80282A28
.long 0x82298029
.long 0xBE801D28
.long 0x8628108F
.long 0x80290CC1
.long 0xBF092902
.long 0x85288028
.long 0xB5280000
.long 0xBF850027
.long 0x862811FF, 0x0000007F
.long 0x80290DC1
.long 0xBF092903
.long 0x85288028
.long 0xB5280000
.long 0xBF850020
.long 0xD1FE0013, 0x020A1F12
.long 0xD3D84018, 0x18000100
.long 0xD3D84019, 0x18000101
.long 0xD3D8401A, 0x18000102
.long 0xD3D8401B, 0x18000103
.long 0xD3D8401C, 0x18000104
.long 0xD3D8401D, 0x18000105
.long 0xD3D8401E, 0x18000106
.long 0xD3D8401F, 0x18000107
.long 0xBF800001
.long 0xBF800000
.long 0xE07C1000, 0x80061813
.long 0xBF800000
.long 0x920AFF1C, 0x00000100
.long 0x80180A18
.long 0x82198019
.long 0xE07C1000, 0x80061C13
.long 0xBF800000
.long 0xBF800000
.long 0xBF820036
.long 0x7E2E02FF, 0x80000000
.long 0xD0C90028, 0x0000210F
.long 0xD0C9002C, 0x00002310
.long 0x86AC2C28
.long 0xD1FE0013, 0x020A1F12
.long 0xD1000013, 0x00B22717
.long 0xD1196A10, 0x00018110
.long 0x9228C01E
.long 0xD1340011, 0x00005111
.long 0x9228C01C
.long 0xD1340012, 0x00005112
.long 0xD0C90028, 0x0000210F
.long 0xD0C9002C, 0x00002310
.long 0x86AC2C28
.long 0xD1FE0016, 0x020A1F12
.long 0xD1000016, 0x00B22D17
.long 0xD3D84018, 0x18000100
.long 0xD3D84019, 0x18000101
.long 0xD3D8401A, 0x18000102
.long 0xD3D8401B, 0x18000103
.long 0xD3D8401C, 0x18000104
.long 0xD3D8401D, 0x18000105
.long 0xD3D8401E, 0x18000106
.long 0xD3D8401F, 0x18000107
.long 0xBF800001
.long 0xBF800000
.long 0xE07C1000, 0x80061813
.long 0xBF800000
.long 0xE07C1000, 0x80061C16
.long 0xBF800000
.long 0xBF800000
.long 0xBF820000
.long 0xBEA81C00
.long 0x812A84FF, 0x00000594
.long 0x80282A28
.long 0x82298029
.long 0xBE801D28
.long 0xB4250000
.long 0xBF840093
.long 0x8628108F
.long 0x80290CC1
.long 0xBF092902
.long 0x85288028
.long 0xB5280000
.long 0xBF85003F
.long 0x862811FF, 0x0000007F
.long 0x80290DC1
.long 0xBF092903
.long 0x85288028
.long 0xB5280000
.long 0xBF850038
.long 0xD1FE0013, 0x02061F12
.long 0xD3D84018, 0x18000100
.long 0xD3D84019, 0x18000101
.long 0xD3D8401A, 0x18000102
.long 0xD3D8401B, 0x18000103
.long 0xD3D8401C, 0x18000104
.long 0xD3D8401D, 0x18000105
.long 0xD3D8401E, 0x18000106
.long 0xD3D8401F, 0x18000107
.long 0xBF800001
.long 0xBF800000
.long 0x0A303024
.long 0x0A323224
.long 0x0A343424
.long 0x0A363624
.long 0x0A383824
.long 0x0A3A3A24
.long 0x0A3C3C24
.long 0x0A3E3E24
.long 0x7E301518
.long 0x7E321519
.long 0xD2A00018, 0x00023318
.long 0x7E34151A
.long 0x7E36151B
.long 0xD2A00019, 0x0002371A
.long 0xE0741000, 0x80061813
.long 0xBF800000
.long 0x7E38151C
.long 0x7E3A151D
.long 0xD2A0001C, 0x00023B1C
.long 0x7E3C151E
.long 0x7E3E151F
.long 0xD2A0001D, 0x00023F1E
.long 0x920AFF1C, 0x00000080
.long 0x80180A18
.long 0x82198019
.long 0xE0741000, 0x80061C13
.long 0xBF800000
.long 0xBF800000
.long 0xBF82011A
.long 0x7E2E02FF, 0x80000000
.long 0xD0C90028, 0x0000210F
.long 0xD0C9002C, 0x00002310
.long 0x86AC2C28
.long 0xD1FE0013, 0x02061F12
.long 0xD1000013, 0x00B22717
.long 0xD1196A10, 0x00018110
.long 0x9228C01E
.long 0xD1340011, 0x00005111
.long 0x9228C01C
.long 0xD1340012, 0x00005112
.long 0xD0C90028, 0x0000210F
.long 0xD0C9002C, 0x00002310
.long 0x86AC2C28
.long 0xD1FE0016, 0x02061F12
.long 0xD1000016, 0x00B22D17
.long 0xD3D84018, 0x18000100
.long 0xD3D84019, 0x18000101
.long 0xD3D8401A, 0x18000102
.long 0xD3D8401B, 0x18000103
.long 0xD3D8401C, 0x18000104
.long 0xD3D8401D, 0x18000105
.long 0xD3D8401E, 0x18000106
.long 0xD3D8401F, 0x18000107
.long 0xBF800001
.long 0xBF800000
.long 0x0A303024
.long 0x0A323224
.long 0x0A343424
.long 0x0A363624
.long 0x0A383824
.long 0x0A3A3A24
.long 0x0A3C3C24
.long 0x0A3E3E24
.long 0x7E301518
.long 0x7E321519
.long 0xD2A00018, 0x00023318
.long 0x7E34151A
.long 0x7E36151B
.long 0xD2A00019, 0x0002371A
.long 0xE0741000, 0x80061813
.long 0xBF800000
.long 0x7E38151C
.long 0x7E3A151D
.long 0xD2A0001C, 0x00023B1C
.long 0x7E3C151E
.long 0x7E3E151F
.long 0xD2A0001D, 0x00023F1E
.long 0xE0741000, 0x80061C16
.long 0xBF800000
.long 0xBF800000
.long 0xBF8200CC
.long 0x8628108F
.long 0x80290CC1
.long 0xBF092902
.long 0x85288028
.long 0xB5280000
.long 0xBF85005B
.long 0x862811FF, 0x0000007F
.long 0x80290DC1
.long 0xBF092903
.long 0x85288028
.long 0xB5280000
.long 0xBF850054
.long 0xD1FE0016, 0x02061F11
.long 0xE0541000, 0x80081816
.long 0x920AFF1E, 0x00000080
.long 0x80200A20
.long 0x82218021
.long 0xE0541000, 0x80081A16
.long 0xD1FE0013, 0x02061F12
.long 0xD3D8401C, 0x18000100
.long 0xD3D8401D, 0x18000101
.long 0xD3D8401E, 0x18000102
.long 0xD3D8401F, 0x18000103
.long 0xD3D84020, 0x18000104
.long 0xD3D84021, 0x18000105
.long 0xD3D84022, 0x18000106
.long 0xD3D84023, 0x18000107
.long 0xBF800001
.long 0xBF800000
.long 0x0A383824
.long 0x0A3A3A24
.long 0x0A3C3C24
.long 0x0A3E3E24
.long 0x0A404024
.long 0x0A424224
.long 0x0A444424
.long 0x0A464624
.long 0xBF8C0F71
.long 0xD3A0001C, 0x14723025
.long 0xD3A0101D, 0x14763025
.long 0xD3A0001E, 0x147A3225
.long 0xD3A0101F, 0x147E3225
.long 0x7E38151C
.long 0x7E3A151D
.long 0xD2A0001C, 0x00023B1C
.long 0x7E3C151E
.long 0x7E3E151F
.long 0xD2A0001D, 0x00023F1E
.long 0xE0741000, 0x80061C13
.long 0xBF800000
.long 0xBF8C0F71
.long 0xD3A00020, 0x14823425
.long 0xD3A01021, 0x14863425
.long 0xD3A00022, 0x148A3625
.long 0xD3A01023, 0x148E3625
.long 0x7E401520
.long 0x7E421521
.long 0xD2A00020, 0x00024320
.long 0x7E441522
.long 0x7E461523
.long 0xD2A00021, 0x00024722
.long 0x920AFF1C, 0x00000080
.long 0x80180A18
.long 0x82198019
.long 0xE0741000, 0x80062013
.long 0xBF800000
.long 0xBF800000
.long 0xBF82006B
.long 0x7E3A02FF, 0x80000000
.long 0xD0C90028, 0x0000210F
.long 0xD0C9002C, 0x00002310
.long 0x86AC2C28
.long 0xD1FE0013, 0x02061F11
.long 0xD1000013, 0x00B2271D
.long 0xE0541000, 0x80081613
.long 0xD1FE0013, 0x02061F12
.long 0xD1000013, 0x00B2271D
.long 0xD1196A10, 0x00018110
.long 0x9228C01E
.long 0xD1340011, 0x00005111
.long 0x9228C01C
.long 0xD1340012, 0x00005112
.long 0xD0C90028, 0x0000210F
.long 0xD0C9002C, 0x00002310
.long 0x86AC2C28
.long 0xD1FE001C, 0x02061F11
.long 0xD100001C, 0x00B2391D
.long 0xE0541000, 0x80081E1C
.long 0xD1FE001C, 0x02061F12
.long 0xD100001C, 0x00B2391D
.long 0xD3D84018, 0x18000100
.long 0xD3D84019, 0x18000101
.long 0xD3D8401A, 0x18000102
.long 0xD3D8401B, 0x18000103
.long 0xD3D84020, 0x18000104
.long 0xD3D84021, 0x18000105
.long 0xD3D84022, 0x18000106
.long 0xD3D84023, 0x18000107
.long 0xBF800001
.long 0xBF800000
.long 0x0A303024
.long 0x0A323224
.long 0x0A343424
.long 0x0A363624
.long 0x0A404024
.long 0x0A424224
.long 0x0A444424
.long 0x0A464624
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
.long 0xE0741000, 0x80061813
.long 0xBF800000
.long 0xD3A00020, 0x14823C25
.long 0xD3A01021, 0x14863C25
.long 0xD3A00022, 0x148A3E25
.long 0xD3A01023, 0x148E3E25
.long 0x7E401520
.long 0x7E421521
.long 0xD2A00020, 0x00024320
.long 0x7E441522
.long 0x7E461523
.long 0xD2A00021, 0x00024722
.long 0xE0741000, 0x8006201C
.long 0xBF800000
.long 0xBF800000
.long 0xBF820000
.long 0xBF810000
