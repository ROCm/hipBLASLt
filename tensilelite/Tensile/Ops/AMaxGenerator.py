################################################################################
#
# Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

from argparse import ArgumentParser
from dataclasses import dataclass
from functools import wraps
from typing import List, Tuple, Optional, Union
from math import log2, log
import os
import yaml
import json
import subprocess
import collections
from contextlib import contextmanager
import Tensile.TensileInstructions as ti
from Tensile.Common import detectGlobalCurrentISA, restoreDefaultGlobalParameters, \
    assignGlobalParameters, getGfxName, gfxArch, globalParameters

def kernel_header(name: str, gfx_arch: str, vgpr: int, sgpr: int, lds: int):
    vgpr = ((vgpr+7)//8)*8
    sgpr = ((sgpr+7)//8)*8
    lds  = ((lds+31)//32)*32

    header = ""
    header += f'.amdgcn_target "amdgcn-amd-amdhsa--{gfx_arch}"\n'
    header += f'.text\n'
    header += f'.protected {name}\n'
    header += f'.globl {name}\n'
    header += f'.p2align 8\n'
    header += f'.type {name},@function\n'
    header += f'.section .rodata,#alloc\n'
    header += f'.p2align 6\n'
    header += f'.amdhsa_kernel {name}\n'
    header += f'  .amdhsa_user_sgpr_kernarg_segment_ptr 1\n'
    if (gfx_arch not in ("gfx900", "gfx908", "gfx1030", "gfx1100", "gfx1101", "gfx1102")):
        header += f'  .amdhsa_accum_offset {vgpr} // accvgpr offset\n'
    header += f'  .amdhsa_next_free_vgpr {vgpr} // vgprs\n'
    header += f'  .amdhsa_next_free_sgpr {sgpr} // sgprs\n'
    header += f'  .amdhsa_group_segment_fixed_size {lds} // lds bytes\n'
    header += f'  .amdhsa_private_segment_fixed_size 0\n'
    header += f'  .amdhsa_system_sgpr_workgroup_id_x 1\n'
    header += f'  .amdhsa_system_sgpr_workgroup_id_y 1\n'
    header += f'  .amdhsa_system_sgpr_workgroup_id_z 1\n'
    header += f'  .amdhsa_system_vgpr_workitem_id 0\n'
    header += f'  .amdhsa_float_denorm_mode_32 3\n'
    header += f'  .amdhsa_float_denorm_mode_16_64 3\n'
    header += f'.end_amdhsa_kernel\n'
    header += f'.text\n'
    return header

@contextmanager
def asm_func(func_name: str, module: ti.Module):
    try:
        module.add(ti.TextBlock(f'{func_name}:\n'))
        yield
    finally:
        end_label_name = f'.L{func_name}_end'
        module.add(ti.SEndpgm())
        module.add(ti.TextBlock(f'{end_label_name}:\n'))
        module.add(ti.TextBlock(f'.size {func_name}, {end_label_name} - {func_name}\n'))

@contextmanager
def asm_loop(mod: ti.Module, name: str, it: str):
    try:
        loop_start_label = ti.Label(name, f'loop {name} starts')
        loop_end_label = ti.Label(f'{name}_end', f'loop {name} ends')
        mod.add(loop_start_label)
        mod.add(ti.SCmpEQU32(ti.sgpr(it), 0))
        mod.add(ti.SCBranchSCC1(loop_end_label.getLabelName()))
        mod.addSpaceLine()
        yield
    finally:
        mod.add(ti.SSubU32(ti.sgpr(it), ti.sgpr(it), 1))
        mod.add(ti.SBranch(loop_start_label.getLabelName()))
        mod.add(loop_end_label)
        mod.addSpaceLine()


class AMaxKernelGenerator:
    srd_num_reg = 4
    srd_alignment = 4

    def __init__(self,
                 i_type: ti.DataType,
                 o_type: ti.DataType,
                 scale_type: ti.DataType,
                 num_workitems: int,
                 wave_size: int,
                 num_load_count: int,
                 num_load_size: int,
                 arch: str,
                 is_scale: bool):
        self.i_type = i_type
        self.o_type = o_type
        self.scale_type = scale_type
        self.bpe = i_type.numBytes()
        self.bpr = i_type.numRegisters()
        self.max_workgoups = num_workitems
        self.num_workitems = num_workitems
        self.wave_size = wave_size
        self.num_load_count = num_load_count
        self.num_load_size = num_load_size
        if is_scale:
            self.sgpr_pool = ti.RegisterPool(64, 's', True)
            self.vgpr_pool = ti.RegisterPool(128, 'v', True)
            self.sgpr_pool.add(0, 63) #TODO: estimate this
            self.vgpr_pool.add(0, 127) #TODO: estimate this
        else:
            self.sgpr_pool = ti.RegisterPool(40, 's', True)
            self.vgpr_pool = ti.RegisterPool(64, 'v', True)
            self.sgpr_pool.add(0, 39) #TODO: estimate this
            self.vgpr_pool.add(0, 63) #TODO: estimate this
        self.debug_label = True
        self.arch = arch
        self.is_scale = is_scale
        self.op = 'AMax'
        self.sgprs  = collections.OrderedDict()
        self.vgprs  = collections.OrderedDict()
        self.label_reduction = ti.Label("reduction", 'reduction')

    @property
    def lds_usage_byte(self) -> int:
        # used in reduce inter wave mean and invvar
        # 4 data * half_wave_num * bpe
        return 4 * (self.num_workitems // 64 // 2) * self.bpe

    @property
    def func_name(self):
        if self.is_scale:
            return f'AMax_Ti_{self.i_type}_To_{self.o_type}_Ts_{self.scale_type}'
        return f'AMax_Ti_{self.i_type}_To_{self.o_type}'

    def dumps(self, format: str) -> str:
        param_dict = {
            'arch': self.arch,
            'op': self.op,
            'is_scale': self.is_scale,
            'func_name': self.func_name,
            'io_type': self.i_type.toChar(),
            'o_type': self.o_type.toChar(),
            'scale_type': self.scale_type.toChar(),
            'num_workitems': self.num_workitems,
        }

        if format.lower() == 'yaml':
            return yaml.dump(param_dict)
        elif format.lower() == 'json':
            return json.dumps(param_dict)
        else:
            assert False, f'Unsupported format {format}'

    def dump(self, format: str, output_path: str):
        s = self.dumps(format)
        with open(output_path, 'w') as f:
            f.write(s)


    def local_read_inst_type(self, num_elements: int, dataType: ti.DataType):
        if dataType.isSingle():
            insts = {
                1: ti.DSLoadB32,
                2: ti.DSLoadB64,
                4: ti.DSLoadB128
            }
        elif dataType.isHalf():
            insts = {
                1: ti.DSLoadU16,
                2: ti.DSLoadB32,
                4: ti.DSLoadB64,
                8: ti.DSLoadB128
            }
        return insts[num_elements]


    def local_write_inst_type(self, num_elements: int, dataType: ti.DataType):
        if dataType.isSingle():
            insts = {
                1: ti.DSStoreB32,
                2: ti.DSStoreB64,
                4: ti.DSStoreB128
            }
        elif dataType.isHalf():
            insts = {
                1: ti.DSStoreB16,
                2: ti.DSStoreB32,
                4: ti.DSStoreB64,
                8: ti.DSStoreB128
            }
        return insts[num_elements]


    def global_read_inst_type(self, num_elements: int, dataType: ti.DataType):
        if dataType.isSingle():
            insts = {
                1: ti.BufferLoadB32,
                2: ti.BufferLoadB64,
                4: ti.BufferLoadB128
            }
        elif dataType.isHalf():
            insts = {
                1: ti.BufferLoadD16B16,
                2: ti.BufferLoadB32,
                4: ti.BufferLoadB64,
                8: ti.BufferLoadB128
            }
        else:
            raise NotImplementedError
        return insts[num_elements]

    def global_write_inst_type(self, num_elements: int, dataType: ti.DataType):
        if dataType.isSingle():
            insts = {
                1: ti.BufferStoreB32,
                2: ti.BufferStoreB64,
                4: ti.BufferStoreB128
            }
        elif dataType.isHalf():
            insts = {
                1: ti.BufferStoreB16,
                2: ti.BufferStoreB32,
                4: ti.BufferStoreB64,
                8: ti.BufferStoreB128
            }
        else:
            raise NotImplementedError
        return insts[num_elements]



    def defineSgpr(self, name, numSgprs, align=1):
        if numSgprs == 0: return
        sgprIdx = self.sgpr_pool.checkOutAligned(numSgprs, align, tag=name, preventOverflow=0)
        self.sgprs[name] = sgprIdx
        return sgprIdx

    def defineVgpr(self, name, numVgprs, align=1):
        if numVgprs == 0: return
        vgprIdx = self.vgpr_pool.checkOutAligned(numVgprs, align, tag=name, preventOverflow=0)
        self.vgprs[name] = vgprIdx
        return vgprIdx


    def kernel_args(self):
        if self.is_scale:
            return (KernelArgument(8,  0, 'global_buffer', 'global'),
                    KernelArgument(8,  8, 'global_buffer', 'global'),
                    KernelArgument(8, 16, 'global_buffer', 'global'),
                    KernelArgument(8, 24, 'global_buffer', 'global'),
                    KernelArgument(8, 32, 'global_buffer', 'global'),
                    KernelArgument(8, 40, 'global_buffer', 'global'),
                    KernelArgument(4, 48, 'by_value'),
                    KernelArgument(4, 52, 'by_value'),
                    KernelArgument(4, 56, 'by_value'),
                    KernelArgument(4, 60, 'by_value'),
                    KernelArgument(4, 64, 'by_value'))
        return (KernelArgument(8,  0, 'global_buffer', 'global'),
                KernelArgument(8,  8, 'global_buffer', 'global'),
                KernelArgument(8, 16, 'global_buffer', 'global'),
                KernelArgument(8, 24, 'global_buffer', 'global'),
                KernelArgument(4, 32, 'by_value'),
                KernelArgument(4, 36, 'by_value'),
                KernelArgument(4, 40, 'by_value'),
                KernelArgument(4, 44, 'by_value'),
                KernelArgument(4, 48, 'by_value'))


    def defineVariables(self):
        self.defineVgpr("Serial",  1, 1)
        self.defineVgpr("Output",  1, 1)
        self.defineVgpr("OutputB", 1, 1)
        self.defineVgpr("Widx",    1, 1)
        self.defineVgpr("Offset",  self.num_load_count, 1)
        self.defineVgpr("Value",   int(self.num_load_count * self.num_load_size * self.bpr), 4)
        if self.is_scale:
            self.defineVgpr("OffsetD", int(self.num_load_count * self.num_load_size * self.bpr), 1)
            self.defineVgpr("OutputD", int(self.num_load_count * self.num_load_size * self.bpr), int(self.num_load_size * self.bpr))
            if self.scale_type == ti.DataType("F8"):
                self.defineVgpr("Fp8NanInf", 1, 1)
                self.defineVgpr("Fp8Max",    1, 1)
                self.defineVgpr("Fp8Max", 1)
                self.defineVgpr("Fp8Min", 1)
                self.defineVgpr("Fp8Tmp", 1)
            elif self.scale_type == ti.DataType("B8"):
                self.defineVgpr("BF8NanInf", 1, 1)
                self.defineVgpr("BF8Max",    1, 1)
                self.defineVgpr("BF8Max", 1)
                self.defineVgpr("BF8Min", 1)
                self.defineVgpr("BF8Tmp", 1)
        self.defineVgpr("Tmp",     4, 1)

        self.defineSgpr("KernelArg", 2)
        self.defineSgpr("WorkGroup0", 1)
        self.defineSgpr("WorkGroup1", 1)
        self.defineSgpr("WorkGroup2", 1)
        self.defineSgpr("AddressOut", 2, 2)
        if self.is_scale:
            self.defineSgpr("AddressOutD", 2, 2)
        self.defineSgpr("AddressIn",  2, 2)
        if self.is_scale:
            self.defineSgpr("AddressScale", 2, 2)
        self.defineSgpr("AddressWk",  2, 2)
        self.defineSgpr("AddressSy",  2, 2)
        self.defineSgpr("SizeLength", 1)
        self.defineSgpr("IsDivided", 1)
        self.defineSgpr("Divided", 1)
        self.defineSgpr("WorkSize",   1)
        self.defineSgpr("NumGroup",   1)

        self.defineSgpr("MainLoop", 1)
        self.defineSgpr("Offset", 1)
        self.defineSgpr("LogWorkSize", 1)
        self.defineSgpr("WGIdx", 1)
        self.defineSgpr("Src", 4, 4)
        self.defineSgpr("Dst", 4, 4)
        if self.is_scale:
            self.defineSgpr("DstD", 4, 4)
            self.defineSgpr("Scale", 1)
            self.defineSgpr("OffsetD", 1)
        self.defineSgpr("Tmp", 6, 2)

        mod = ti.Module("defineVariables")

        for vkey in self.vgprs:
            mod.add(ti.RegSet("v", "vgpr"+vkey, self.vgprs[vkey]))
        mod.addSpaceLine()

        for skey in self.sgprs:
            mod.add(ti.RegSet("s", "sgpr"+skey, self.sgprs[skey]))
        mod.addSpaceLine()

        mod.add(ti.ValueSet("Srd127_96", "0x00020000", format=-1))
        mod.addSpaceLine()
        mod.addSpaceLine()
        return mod


    def load_kernel_args(self):
        mod = ti.Module('Load kernel args')
        mod.addComment0('Load kernel args')
        offset = 0
        if self.is_scale:
            mod.add(ti.SLoadB64(ti.sgpr("AddressOut", 2),    ti.sgpr("KernelArg", 2),  0))
            mod.add(ti.SLoadB64(ti.sgpr("AddressOutD", 2),   ti.sgpr("KernelArg", 2),  8))
            mod.add(ti.SLoadB64(ti.sgpr("AddressIn", 2),     ti.sgpr("KernelArg", 2),  16))
            mod.add(ti.SLoadB64(ti.sgpr("AddressScale", 2),  ti.sgpr("KernelArg", 2),  24))
            mod.add(ti.SLoadB64(ti.sgpr("AddressWk", 2),     ti.sgpr("KernelArg", 2),  32))
            mod.add(ti.SLoadB64(ti.sgpr("AddressSy", 2),     ti.sgpr("KernelArg", 2),  40))
            mod.add(ti.SLoadB32(ti.sgpr("SizeLength"),       ti.sgpr("KernelArg", 2),  48))
            mod.add(ti.SLoadB32(ti.sgpr("IsDivided"),        ti.sgpr("KernelArg", 2),  52))
            mod.add(ti.SLoadB32(ti.sgpr("Divided"),          ti.sgpr("KernelArg", 2),  56))
            mod.add(ti.SLoadB32(ti.sgpr("WorkSize"),         ti.sgpr("KernelArg", 2),  60))
            mod.add(ti.SLoadB32(ti.sgpr("NumGroup"),         ti.sgpr("KernelArg", 2),  64))
        else:
            mod.add(ti.SLoadB64(ti.sgpr("AddressOut", 2),    ti.sgpr("KernelArg", 2),   0))
            mod.add(ti.SLoadB64(ti.sgpr("AddressIn", 2),     ti.sgpr("KernelArg", 2),   8))
            mod.add(ti.SLoadB64(ti.sgpr("AddressWk", 2),     ti.sgpr("KernelArg", 2),  16))
            mod.add(ti.SLoadB64(ti.sgpr("AddressSy", 2),     ti.sgpr("KernelArg", 2),  24))
            mod.add(ti.SLoadB32(ti.sgpr("SizeLength"),       ti.sgpr("KernelArg", 2),  32))
            mod.add(ti.SLoadB32(ti.sgpr("IsDivided"),        ti.sgpr("KernelArg", 2),  36))
            mod.add(ti.SLoadB32(ti.sgpr("Divided"),          ti.sgpr("KernelArg", 2),  40))
            mod.add(ti.SLoadB32(ti.sgpr("WorkSize"),         ti.sgpr("KernelArg", 2),  44))
            mod.add(ti.SLoadB32(ti.sgpr("NumGroup"),         ti.sgpr("KernelArg", 2),  48))

        mod.add(ti.SWaitCnt(lgkmcnt=0))
        mod.addSpaceLine()

        mod.add(ti.VMovB32(ti.vgpr("Tmp"), ti.sgpr("WorkSize")))
        mod.add(ti.VCvtU32toF32(ti.vgpr("Tmp"), ti.vgpr("Tmp")))
        mod.add(ti.SNop(0))
        mod.add(ti.VLogF32(ti.vgpr("Tmp"), ti.vgpr("Tmp")))
        mod.add(ti.SNop(0))
        mod.add(ti.VCvtF32toU32(ti.vgpr("Tmp"), ti.vgpr("Tmp")))
        mod.add(ti.SNop(0))
        mod.add(ti.VReadfirstlaneB32(ti.sgpr("LogWorkSize"), ti.vgpr("Tmp")))

        mod.addSpaceLine()
        mod.addSpaceLine()
        return mod


    def init_param(self) -> ti.Module:
        mod = ti.Module("init_param")
        mod.addComment0("init_param")
        mod.add(ti.SLShiftLeftB32(ti.sgpr("Tmp"), int(log2(self.bpe)), ti.sgpr("WorkSize")))
        mod.add(ti.SMulI32(ti.sgpr("Tmp"), ti.sgpr("WorkGroup0"), ti.sgpr("Tmp")))
        mod.addSpaceLine()

        mod.add(ti.SAddU32(ti.sgpr("Src+0"), ti.sgpr("AddressIn+0"), ti.sgpr("Tmp")))
        mod.add(ti.SAddCU32(ti.sgpr("Src+1"), ti.sgpr("AddressIn+1"), 0))
        mod.add(ti.SLShiftLeftB32(ti.sgpr("Src+2"), int(log2(self.bpe)), ti.sgpr("SizeLength")))
        mod.add(ti.SSubU32(ti.sgpr("Src+2"), ti.sgpr("Src+2"),ti.sgpr("Tmp")))
        mod.add(ti.SMovB32(ti.sgpr("Src+3"), "Srd127_96"))
        mod.addSpaceLine()

        mod.add(ti.VMovB32(ti.vgpr("Output"), 0))
        mod.add(ti.SMovB32(ti.sgpr("WGIdx"),ti.sgpr("WorkGroup0")))
        mod.addSpaceLine()

        if self.is_scale: # init inputScale
            mod.add(ti.SLoadB32(ti.sgpr("Scale"), ti.sgpr("AddressScale", 2), 0))
            mod.add(ti.SWaitCnt(lgkmcnt=0))
            mod.addSpaceLine()

            if self.scale_type == ti.DataType("F8"):
                mod.add(ti.VMovB32(ti.vgpr("Fp8NanInf"), "0x207", "Nan and +/- inf"))
                mod.add(ti.VMovB32(ti.vgpr("Fp8Max"), "0x43700000", "Fp8 Max value 240 as float32"))
                mod.add(ti.VMovB32(ti.vgpr("Fp8Min"), "0xc3700000", "Fp8 Min value -240 as float32"))
            elif self.scale_type == ti.DataType("B8"):
                mod.add(ti.VMovB32(ti.vgpr("BF8NanInf"), "0x207", "Nan and +/- inf"))
                mod.add(ti.VMovB32(ti.vgpr("BF8Max"), "0x47600000", "BF8 Max value 57344 as float32"))
                mod.add(ti.VMovB32(ti.vgpr("BF8Min"), "0xc7600000", "BF8 Min value -57344 as float32"))

            mod.addSpaceLine()

            mod.add(ti.SMulI32(ti.sgpr("Tmp"), ti.sgpr("WorkGroup0"), ti.sgpr("WorkSize")))

            mod.add(ti.SAddU32(ti.sgpr("DstD+0"), ti.sgpr("AddressOutD+0"), ti.sgpr("Tmp")))
            mod.add(ti.SAddCU32(ti.sgpr("DstD+1"), ti.sgpr("AddressOutD+1"), 0))
            mod.add(ti.SSubU32(ti.sgpr("DstD+2"), ti.sgpr("SizeLength"), ti.sgpr("Tmp")))
            mod.add(ti.SMovB32(ti.sgpr("DstD+3"), "Srd127_96"))
            for i in range(int(self.num_load_count * self.num_load_size * self.bpr)):
                mod.add(ti.VMovB32(ti.vgpr(f"OutputD+{i}"), 0))
            mod.addSpaceLine()

        mod.addSpaceLine()

        return mod


    def calculate_global_address(self) -> ti.Module:
        mod = ti.Module("calculate_global_address")
        mod.addComment0("calculate_global_address")

        # offset for buffer load
        mod.add(ti.VLShiftRightB32(ti.vgpr("Widx"), int(log2(self.wave_size)), ti.vgpr("Serial")))
        mod.add(ti.VAndB32(ti.vgpr("Offset+0"), self.wave_size-1, ti.vgpr("Serial")))
        mod.add(ti.VLShiftLeftB32(ti.vgpr("Offset+0"), int(log2(self.num_load_size * self.bpe)), ti.vgpr("Offset+0")))
        mod.add(ti.SMovB32(ti.sgpr("Tmp"), self.wave_size * self.num_load_size * self.bpe))
        for i in range(0, self.num_load_count-1):
            mod.add(ti.VAddU32(ti.vgpr(f"Offset+{i+1}"), ti.vgpr(f"Offset+{i}"), ti.sgpr("Tmp")))
        mod.addSpaceLine()

        offset = self.num_load_count * self.wave_size * self.num_load_size * self.bpe
        mod.add(ti.VLShiftLeftB32(ti.vgpr("Tmp"), int(log2(offset)), ti.vgpr("Widx")))
        for i in range(0, self.num_load_count):
            mod.add(ti.VAddU32(ti.vgpr(f"Offset+{i}"), ti.vgpr(f"Offset+{i}"), ti.vgpr("Tmp")))
        mod.addSpaceLine()

        if self.is_scale: # offset for buffer store
            mod.add(ti.VLShiftRightB32(ti.vgpr("Widx"), int(log2(self.wave_size)), ti.vgpr("Serial")))
            mod.add(ti.VAndB32(ti.vgpr("OffsetD+0"), self.wave_size-1, ti.vgpr("Serial")))
            mod.add(ti.VLShiftLeftB32(ti.vgpr("OffsetD+0"), int(log2(self.num_load_size * self.scale_type.numBytes())), ti.vgpr("OffsetD+0")))
            for i in range(1, self.num_load_size):
                mod.add(ti.VAddU32(ti.vgpr(f"OffsetD+{i}"), ti.vgpr(f"OffsetD+{i-1}"), self.scale_type.numBytes()))
            mod.add(ti.SMovB32(ti.sgpr("Tmp"), self.wave_size * self.num_load_size * self.scale_type.numBytes()))
            for i in range(1, self.num_load_count):
                for j in range(0, self.num_load_size):
                    mod.add(ti.VAddU32(ti.vgpr(f"OffsetD+{j+i*self.num_load_size}"), ti.vgpr(f"OffsetD+{j+(i-1)*self.num_load_size}"), ti.sgpr("Tmp")))
            mod.addSpaceLine()

            offset = self.wave_size * self.num_load_count * self.num_load_size *self.scale_type.numBytes()
            mod.add(ti.VLShiftLeftB32(ti.vgpr("Tmp"), int(log2(offset)), ti.vgpr("Widx")))
            for i in range(self.num_load_count * self.num_load_size):
                mod.add(ti.VAddU32(ti.vgpr(f"OffsetD+{i}"), ti.vgpr(f"OffsetD+{i}"), ti.vgpr("Tmp")))
        mod.addSpaceLine()

        return mod


    def block_max(self, wait, last=False) -> ti.Module:
        vgprs = int(self.num_load_size * self.bpr)
        buffer_load = self.global_read_inst_type(self.num_load_size, self.i_type)
        vgpr_per_load = int(self.num_load_size * self.bpr)

        mod = ti.Module("block_max")
        mod.addComment0("block_max")

        for i in range(0, self.num_load_count):
            mod.add(ti.SWaitCnt(vmcnt=wait))
            for j in range(0, vgprs):
                mod.add(self.max_per_data(i*vgprs+j, 1 / self.i_type.numRegisters()))
                if self.is_scale:
                    mod.add(self.scale_per_data(i*vgprs+j))
                    mod.add(ti.BufferStoreB8(ti.vgpr(f"OutputD+{i*vgprs+j}"), \
                                             ti.vgpr(f"OffsetD+{i*vgprs+j}"), \
                                             ti.sgpr("DstD",4), ti.sgpr("OffsetD"), ti.MUBUFModifiers(offen=True)))
            if last:
                wait = wait - 1
            else:
                mod.add(buffer_load(ti.vgpr(f"Value+{i*vgpr_per_load}", int(vgpr_per_load)), ti.vgpr(f"Offset+{i}"), ti.sgpr("Src",4),  ti.sgpr("Offset"), ti.MUBUFModifiers(offen=True)))

        mod.addSpaceLine()
        return mod


    def sum_per_blocksize(self) -> ti.Module:
        mod = ti.Module("sum_per_blocksize")
        mod.addComment0("sum_per_blocksize")

        label_sum_per_blocksize = ti.Label("sum_per_blocksize", 'sum_per_blocksize')
        label_loop = ti.Label("loop", 'loop')
        label_last_loop = ti.Label("last_loop", 'last_loop')
        label_loop_end = ti.Label("loop_end", 'loop_end')
        label_sum_per_blocksize_end = ti.Label("sum_per_blocksize_end", 'sum_per_blocksize_end')

        mod.add(ti.SLShiftRightB32(ti.sgpr("MainLoop"), ti.sgpr("LogWorkSize"), ti.sgpr("SizeLength")))
        mod.add(label_sum_per_blocksize)
        mod.add(ti.SCmpGeI32(ti.sgpr("WGIdx"), ti.sgpr("MainLoop")))
        mod.add(ti.SCBranchSCC1(label_sum_per_blocksize_end.getLabelName()))
        mod.addSpaceLine()

        if self.is_scale:
            mod.add(ti.SMovB32(ti.sgpr("OffsetD"), 0))
        mod.add(ti.SMovB32(ti.sgpr("Offset"), 0))
        mod.add(ti.SLShiftLeftB32(ti.sgpr("Tmp"), 1, ti.sgpr("WorkSize")))
        mod.add(ti.SCmpGeI32(ti.sgpr("Offset"), ti.sgpr("Tmp")))
        mod.add(ti.SCBranchSCC1(label_loop_end.getLabelName()))
        mod.addSpaceLine()

        buffer_load = self.global_read_inst_type(self.num_load_size, self.i_type)
        vgpr_per_load = int(self.num_load_size * self.bpr)
        for i in range(0, self.num_load_count):
            mod.add(buffer_load(ti.vgpr(f"Value+{i*vgpr_per_load}", int(vgpr_per_load)), ti.vgpr(f"Offset+{i}"), ti.sgpr("Src",4), ti.sgpr("Offset"), ti.MUBUFModifiers(offen=True)))
        mod.addSpaceLine()

        offset = self.num_load_count * self.num_workitems * self.num_load_size * self.bpe
        mod.add(ti.SAddI32(ti.sgpr("Offset"), ti.sgpr("Offset"), offset))
        mod.addSpaceLine()

        mod.add(label_loop)
        mod.addSpaceLine()

        mod.add(ti.SLShiftLeftB32(ti.sgpr("Tmp"), int(log2(self.bpe)), ti.sgpr("WorkSize")))
        mod.add(ti.SCmpGeI32(ti.sgpr("Offset"), ti.sgpr("Tmp")))
        mod.add(ti.SCBranchSCC1(label_last_loop.getLabelName()))
        mod.addSpaceLine()

        mod.add(self.block_max(self.num_load_count-1, False))

        if self.is_scale:
            offsetD = self.num_load_count * self.num_workitems * self.num_load_size * self.scale_type.numBytes()
            mod.add(ti.SAddI32(ti.sgpr("OffsetD"), ti.sgpr("OffsetD"), offsetD))
        mod.add(ti.SAddI32(ti.sgpr("Offset"), ti.sgpr("Offset"), offset))
        mod.add(ti.SBranch(label_loop.getLabelName()))
        mod.addSpaceLine()

        mod.add(label_last_loop)
        mod.addSpaceLine()

        mod.add(self.block_max(self.num_load_count-1, True))
        mod.addSpaceLine()

        mod.add(label_loop_end)
        mod.addSpaceLine()

        mod.add(ti.SLShiftLeftB32(ti.sgpr("Tmp"), int(log2(self.bpe)), ti.sgpr("WorkSize")))
        mod.add(ti.SMulI32(ti.sgpr("Tmp"), ti.sgpr("Tmp"), ti.sgpr("NumGroup")))
        for i in range(0, self.num_load_count):
           mod.add(ti.VAddU32(ti.vgpr(f"Offset+{i}"), ti.vgpr(f"Offset+{i}"), ti.sgpr("Tmp")))
        mod.addSpaceLine()

        if self.is_scale:
            mod.add(ti.SLShiftLeftB32(ti.sgpr("Tmp"), int(log2(self.scale_type.numBytes())), ti.sgpr("WorkSize")))
            mod.add(ti.SMulI32(ti.sgpr("Tmp"), ti.sgpr("Tmp"), ti.sgpr("NumGroup")))
            for i in range(0, self.num_load_count * self.num_load_size):
               mod.add(ti.VAddU32(ti.vgpr(f"OffsetD+{i}"), ti.vgpr(f"OffsetD+{i}"), ti.sgpr("Tmp")))
            mod.addSpaceLine()

        mod.add(ti.SAddI32( ti.sgpr("WGIdx"), ti.sgpr("WGIdx"), ti.sgpr("NumGroup")))
        mod.add(ti.SBranch(label_sum_per_blocksize.getLabelName()))
        mod.addSpaceLine()

        mod.add(label_sum_per_blocksize_end)
        mod.addSpaceLine()

        mod.addSpaceLine()

        return mod

    def adjust_global_address(self) -> ti.Module:
        mod = ti.Module("adjust_global_address")
        mod.addComment0("adjust_global_address")

        mod.add(ti.SSubU32(ti.sgpr("Tmp"), ti.sgpr("WorkSize"), 1))
        mod.add(ti.SAndB32(ti.sgpr("Tmp"), ti.sgpr("SizeLength"), ti.sgpr("Tmp")))
        mod.add(ti.SCmpEQU32(ti.sgpr("Tmp"), 0))
        mod.add(ti.SCBranchSCC1(self.label_reduction.getLabelName()))
        mod.addSpaceLine()

        mod.add(ti.SLShiftRightB32(ti.sgpr("Tmp"), ti.sgpr("LogWorkSize"), ti.sgpr("SizeLength")))
        mod.add(ti.SCmpEQU32(ti.sgpr("WGIdx"), ti.sgpr("Tmp")))
        mod.add(ti.SCBranchSCC0(self.label_reduction.getLabelName()))
        mod.addSpaceLine()

        mod.add(ti.SLShiftRightB32(ti.sgpr("Tmp"), ti.sgpr("LogWorkSize"), ti.sgpr("SizeLength")))
        mod.add(ti.SAddU32(ti.sgpr("Tmp+1"), ti.sgpr("LogWorkSize"), int(log2(self.bpe))))
        mod.add(ti.SLShiftLeftB32(ti.sgpr("Tmp"), ti.sgpr("Tmp+1"), ti.sgpr("Tmp")))
        mod.addSpaceLine()

        mod.add(ti.SAddU32(ti.sgpr("Src+0"), ti.sgpr("AddressIn+0"), ti.sgpr("Tmp")))
        mod.add(ti.SAddCU32(ti.sgpr("Src+1"), ti.sgpr("AddressIn+1"), 0))
        mod.add(ti.SLShiftLeftB32(ti.sgpr("Src+2"), int(log2(self.bpe)), ti.sgpr("SizeLength")))
        mod.add(ti.SSubU32(ti.sgpr("Src+2"), ti.sgpr("Src+2"), ti.sgpr("Tmp")))
        mod.add(ti.SMovB32(ti.sgpr("Src+3"), "Srd127_96"))
        mod.addSpaceLine()


        mod.add(ti.VLShiftLeftB32(ti.vgpr("Offset"), int(log2(self.num_load_size * self.bpe)), ti.vgpr("Serial")))
        mod.add(ti.SMovB32(ti.sgpr("Tmp"), self.num_workitems * self.num_load_size * self.bpe))
        for i in range(0, 3):
            mod.add(ti.VAddU32(ti.vgpr(f"Offset+{i+1}"), ti.vgpr(f"Offset+{i}"), ti.sgpr("Tmp")))
        mod.addSpaceLine()

        if self.is_scale:
            mod.add(ti.SLShiftRightB32(ti.sgpr("Tmp"), ti.sgpr("LogWorkSize"), ti.sgpr("SizeLength")))
            mod.add(ti.SAddU32(ti.sgpr("Tmp+1"), ti.sgpr("LogWorkSize"), int(log2(self.scale_type.numBytes()))))
            mod.add(ti.SLShiftLeftB32(ti.sgpr("Tmp"), ti.sgpr("Tmp+1"), ti.sgpr("Tmp")))
            mod.addSpaceLine()

            mod.add(ti.SAddU32(ti.sgpr("DstD+0"), ti.sgpr("AddressOutD+0"), ti.sgpr("Tmp")))
            mod.add(ti.SAddCU32(ti.sgpr("DstD+1"), ti.sgpr("AddressOutD+1"), 0))
            mod.add(ti.SLShiftLeftB32(ti.sgpr("DstD+2"), int(log2(self.scale_type.numBytes())), ti.sgpr("SizeLength")))
            mod.add(ti.SSubU32(ti.sgpr("DstD+2"), ti.sgpr("DstD+2"), ti.sgpr("Tmp")))
            mod.add(ti.SMovB32(ti.sgpr("DstD+3"), "Srd127_96"))
            mod.addSpaceLine()

            mod.add(ti.VLShiftLeftB32(ti.vgpr("OffsetD"), int(log2(self.num_load_size * self.scale_type.numBytes())), ti.vgpr("Serial")))
            for i in range(0, self.num_load_size-1):
                mod.add(ti.VAddU32(ti.vgpr(f"OffsetD+{i+1}"), ti.vgpr(f"OffsetD+{i}"), self.scale_type.numBytes()))
            mod.add(ti.SMovB32(ti.sgpr("Tmp"), self.num_workitems * self.num_load_size * self.scale_type.numBytes()))
            for i in range(1, 4):
                for j in range(0, self.num_load_size):
                    mod.add(ti.VAddU32(ti.vgpr(f"OffsetD+{i*self.num_load_size+j}"), ti.vgpr(f"OffsetD+{(i-1)*self.num_load_size+j}"), ti.sgpr("Tmp")))

            mod.addSpaceLine()

        mod.add(ti.SSubU32(ti.sgpr("Tmp"), ti.sgpr("WorkSize"), 1))
        mod.add(ti.SAndB32(ti.sgpr("SizeLength"), ti.sgpr("SizeLength"), ti.sgpr("Tmp")))

        mod.addSpaceLine()
        return mod


    def max_per_data(self, i, numElement) -> ti.Module:
        mod = ti.Module("max_per_data")
        if (self.i_type.isHalf()):
            mod.add(ti.VMaxF16(ti.vgpr("Output"), ti.vgpr("Output"), ti.SrcAbs(ti.vgpr(f"Value+{i}"))))
            if numElement == 2:
                mod.add(ti.VLShiftRightB32(ti.vgpr(f"Value+{i}"), 16, ti.vgpr(f"Value+{i}")))
                mod.add(ti.VMaxF16(ti.vgpr("Output"), ti.vgpr("Output"), ti.SrcAbs(ti.vgpr(f"Value+{i}"))))
        elif (self.i_type.isSingle()):
            mod.add(ti.VMaxF32(ti.vgpr("Output"), ti.vgpr("Output"), ti.SrcAbs(ti.vgpr(f"Value+{i}"))))
        return mod


    def scale_per_data(self, i) -> ti.Module:
        mod = ti.Module("scale_per_data")
        mod.add(ti.VMulF32(ti.vgpr(f"OutputD+{i}"), ti.sgpr("Scale"), ti.vgpr(f"Value+{i}")))
        if self.scale_type == ti.DataType("F8"):
            mod.add(ti.VCmpClassF32(dst="vcc", src0=ti.vgpr(f"OutputD+{i}"), src1=ti.vgpr("Fp8NanInf")))
            mod.add(ti.VMed3F32(dst=ti.vgpr("Fp8Tmp"), src0=ti.vgpr(f"OutputD+{i}"), src1=ti.vgpr("Fp8Min"), src2=ti.vgpr("Fp8Max")))
            mod.add(ti.VCndMaskB32(dst=ti.vgpr(f"OutputD+{i}"), src0=ti.vgpr("Fp8Tmp"), src1=ti.vgpr(f"OutputD+{i}"), src2="vcc"))
            mod.add(ti.VCvtPkF32toFP8(ti.vgpr(f"OutputD+{i}"), ti.vgpr(f"OutputD+{i}"), ti.vgpr(f"OutputD+{i}")))
        elif self.scale_type == ti.DataType("B8"):
            mod.add(ti.VCmpClassF32(dst="vcc", src0=ti.vgpr(f"OutputD+{i}"), src1=ti.vgpr("BF8NanInf")))
            mod.add(ti.VMed3F32(dst=ti.vgpr("BF8Tmp"), src0=ti.vgpr(f"OutputD+{i}"), src1=ti.vgpr("BF8Min"), src2=ti.vgpr("BF8Max")))
            mod.add(ti.VCndMaskB32(dst=ti.vgpr(f"OutputD+{i}"), src0=ti.vgpr("BF8Tmp"), src1=ti.vgpr(f"OutputD+{i}"), src2="vcc"))
            mod.add(ti.VCvtPkF32toBF8(ti.vgpr(f"OutputD+{i}"), ti.vgpr(f"OutputD+{i}"), ti.vgpr(f"OutputD+{i}")))
        return mod


    def sum_per_threadx4x4(self) -> ti.Module:
        mod = ti.Module("sum_per_threadxdwordx4x4")
        mod.addComment0("sum_per_threadxdwordx4x4")
        mod.add(ti.SLShiftRightB32(ti.sgpr("MainLoop"), \
                                   int(log2(self.num_workitems * 4 * self.num_load_size)), \
                                   ti.sgpr("SizeLength")))
        with asm_loop(mod, "sum_per_threadx4x4", "MainLoop"):
            num_load_registers = int(self.num_load_size * self.bpr)
            for i in range(0, 4): # unroll
                mod.add(ti.BufferLoadB128(ti.vgpr(f"Value+{i*num_load_registers}",4), \
                                          ti.vgpr(f"Offset+{i}"), ti.sgpr("Src",4), 0, ti.MUBUFModifiers(offen=True)))
            mod.addSpaceLine()
            # max operation
            for i in range(0, 4): # unroll
                mod.add(ti.SWaitCnt(vmcnt=(4-i-1)))
                for j in range(0, num_load_registers): # dwordx4
                    index = i * num_load_registers + j
                    mod.add(self.max_per_data(index, 1 / self.bpr))
                    if self.is_scale:
                        mod.add(self.scale_per_data(index))
                        mod.add(ti.BufferStoreB8(ti.vgpr(f"OutputD+{i*num_load_registers+j}"), \
                                                 ti.vgpr(f"OffsetD+{i*num_load_registers+j}"), \
                                                 ti.sgpr("DstD",4), 0, ti.MUBUFModifiers(offen=True)))
                        mod.addSpaceLine()
            mod.addSpaceLine()


            # adjust offset of buffer load
            # total bytes = num_workitems * num_unroll * load_size_in_bytes
            # num_unroll = num_load_count
            # load_size_in_bytes = dwordx4 = num_load_size * 4
            mod.add(ti.SMovB32(ti.sgpr("Tmp"), self.num_workitems * 4 * self.num_load_size * self.bpe))
            for i in range(0, 4):
                mod.add(ti.VAddU32(ti.vgpr(f"Offset+{i}"), ti.vgpr(f"Offset+{i}"), ti.sgpr("Tmp")))
            mod.addSpaceLine()

            if self.is_scale: # adjust offset of buffer store fp8
                mod.add(ti.SMovB32(ti.sgpr("Tmp"), self.num_workitems * 4 * 4))
                for i in range(0, 4):
                    for j in range(0, self.num_load_size):
                        mod.add(ti.VAddU32(ti.vgpr(f"OffsetD+{i*self.num_load_size+j}"), \
                                           ti.vgpr(f"OffsetD+{i*self.num_load_size+j}"), ti.sgpr("Tmp")))
                mod.addSpaceLine()
        mod.addSpaceLine()
        return mod


    def sum_per_threadx4(self) -> ti.Module:
        mod = ti.Module("sum_per_threadx4")
        mod.addComment0("sum_per_threadx4")
        mod.add(ti.SLShiftRightB32(ti.sgpr("MainLoop"), \
                                   int(log2(self.num_workitems * self.num_load_size)), \
                                   ti.sgpr("SizeLength")))
        mod.add(ti.SAndB32(ti.sgpr("MainLoop"), 4-1, ti.sgpr("MainLoop")))
        with asm_loop(mod, "sum_per_threadx4", "MainLoop"):
            num_regsters_per_load = int(self.num_load_size * self.bpr)
            mod.add(ti.BufferLoadB128(ti.vgpr("Value",4), ti.vgpr("Offset"), ti.sgpr("Src",4), 0, ti.MUBUFModifiers(offen=True)))
            mod.addSpaceLine()
            mod.add(ti.SWaitCnt(vmcnt=0))
            # max operation
            for i in range(0, num_regsters_per_load): # dwordx4
                mod.add(self.max_per_data(i, 1 / self.i_type.numRegisters()))
                if self.is_scale: # buffer store fp8
                    mod.add(self.scale_per_data(i))
                    mod.add(ti.BufferStoreB8(ti.vgpr(f"OutputD+{i}"), \
                                             ti.vgpr(f"OffsetD+{i}"), \
                                             ti.sgpr("DstD",4), 0, ti.MUBUFModifiers(offen=True)))
            mod.addSpaceLine()
            # adjust offset of buffer load
            mod.add(ti.SMovB32(ti.sgpr("Tmp"), self.num_workitems * self.num_load_size * self.bpe))
            mod.add(ti.VAddU32(ti.vgpr("Offset"), ti.vgpr("Offset"), ti.sgpr("Tmp")))
            if self.is_scale: # adjust offset of buffer store fp8
                mod.add(ti.SMovB32(ti.sgpr("Tmp"), self.num_workitems * 4))
                for i in range(0, self.num_load_size):
                    mod.add(ti.VAddU32(ti.vgpr(f"OffsetD+{i}"), \
                                       ti.vgpr(f"OffsetD+{i}"), ti.sgpr("Tmp")))
                mod.addSpaceLine()
        mod.addSpaceLine()
        return mod


    def adjust_global_address_2(self) -> ti.Module:
        mod = ti.Module("adjust_global_address_2")
        mod.addComment0("adjust_global_address_2")

        # adjust buffer load offset
        #    buffer_load_dwordx4 = byte * 4 * 4
        # -) buffer_load_dword   = byte * 1 * 4
        # --------------------------------------
        #                          byte * 3 * 4
        mod.add(ti.VMulLOU32(ti.vgpr("Tmp"), 3, ti.vgpr("Serial")))
        mod.add(ti.VLShiftLeftB32(ti.vgpr("Tmp"), 2, ti.vgpr("Tmp")))
        mod.add(ti.VSubU32(ti.vgpr("Offset"), ti.vgpr("Offset"), ti.vgpr("Tmp")))
        if (self.i_type.isHalf()):
            mod.add(ti.VLShiftLeftB32(ti.vgpr("Tmp"), 1, ti.vgpr("Serial")))
            mod.add(ti.VSubU32(ti.vgpr("Offset"), ti.vgpr("Offset"), ti.vgpr("Tmp")))
        mod.addSpaceLine()

        if self.is_scale: # adjust buffer store offset
            mod.add(ti.VMulLOU32(ti.vgpr("Tmp"), 3, ti.vgpr("Serial")))
            mod.add(ti.VSubU32(ti.vgpr("OffsetD"), ti.vgpr("OffsetD"), ti.vgpr("Tmp")))

        mod.addSpaceLine()
        mod.addSpaceLine()
        return mod


    def sum_per_thread(self) -> ti.Module:
        offset = self.num_workitems
        mod = ti.Module("sum_per_thread")
        mod.addComment0("sum_per_thread")
        mod.add(ti.SLShiftRightB32(ti.sgpr("MainLoop"), int(log2(offset)), ti.sgpr("SizeLength")))
        mod.add(ti.SAndB32(ti.sgpr("MainLoop"), ti.sgpr("MainLoop"), self.num_load_size - 1))
        with asm_loop(mod, "sum_per_thread", "MainLoop"):
            BufferLoadx1 = self.global_read_inst_type(1, self.i_type)
            mod.add(BufferLoadx1(ti.vgpr("Value"), ti.vgpr("Offset"), ti.sgpr("Src",4), 0, ti.MUBUFModifiers(offen=True)))
            mod.add(ti.SWaitCnt(vmcnt=0))
            mod.addSpaceLine()
            mod.add(self.max_per_data(0, 1))
            if self.is_scale:
                mod.add(self.scale_per_data(0))
                mod.add(ti.BufferStoreB8(ti.vgpr("OutputD"),
                                         ti.vgpr("OffsetD"), ti.sgpr("DstD",4), 0, ti.MUBUFModifiers(offen=True)))
                mod.addSpaceLine()
            mod.add(ti.SMovB32(ti.sgpr("Tmp"), self.num_workitems * self.bpe))
            mod.add(ti.VAddU32(ti.vgpr("Offset"), ti.vgpr("Offset"), ti.sgpr("Tmp")))
            mod.addSpaceLine()
            if self.is_scale:
                mod.add(ti.SMovB32(ti.sgpr("Tmp"), self.num_workitems))
                mod.add(ti.VAddU32(ti.vgpr("OffsetD"), ti.vgpr("OffsetD"), ti.sgpr("Tmp")))
                mod.addSpaceLine()
        return mod


    def sum_in_some_thread(self)  -> ti.Module:
        label_sum_end = ti.Label("sum", f'loop sum end')
        mod = ti.Module("sum_in_some_thread")
        mod.addComment0("sum_in_some_thread")
        mod.add(ti.SAndB32(ti.sgpr("MainLoop"), ti.sgpr("SizeLength"), self.num_workitems-1))
        mod.add(ti.VCmpLtU32("vcc", ti.vgpr("Serial"), ti.sgpr("MainLoop")))
        mod.add(ti.SCBranchVCCZ(label_sum_end.getLabelName()))
        mod.add(ti.SMovB64("exec", "vcc"))
        mod.add(ti.SNop(1))
        BufferLoadx1 = self.global_read_inst_type(1, self.i_type)
        mod.add(BufferLoadx1(ti.vgpr("Value"), ti.vgpr("Offset"), ti.sgpr("Src",4), 0, ti.MUBUFModifiers(offen=True)))
        mod.add(ti.SWaitCnt(vmcnt=0))
        mod.addSpaceLine()
        mod.add(self.max_per_data(0, 1))
        if self.is_scale:
            mod.add(self.scale_per_data(0))
            mod.add(ti.BufferStoreB8(ti.vgpr("OutputD"),
                                     ti.vgpr("OffsetD"), ti.sgpr("DstD",4), 0, ti.MUBUFModifiers(offen=True)))
            mod.addSpaceLine()
        mod.add(ti.SMovB64("exec", "-1"))
        mod.add(ti.SNop(1))
        mod.add(label_sum_end)
        mod.addSpaceLine()
        mod.addSpaceLine()
        return mod


    def merge_sum(self) -> ti.Module:
        mod = ti.Module("merge_sum")
        if (self.i_type.isHalf()):
            mod.add(ti.VMaxF16(ti.vgpr("Output"), ti.vgpr("Output"), ti.vgpr("OutputB")))
        elif (self.i_type.isSingle()):
            mod.add(ti.VMaxF32(ti.vgpr("Output"), ti.vgpr("Output"), ti.vgpr("OutputB")))

        return mod


    def intra_wave_reduction(self, postfix) -> ti.Module:
        label = ti.Label(f"permute_{postfix}", f"permute_{postfix}")
        mod = ti.Module("intra_wave_reduction")
        mod.addComment0("intra_wave_reduction")
        mod.add(ti.SMovB32(ti.sgpr("Tmp"), 1))
        mod.add(label)
        mod.addSpaceLine()
        mod.add(ti.VAddU32(ti.vgpr("Tmp"), ti.sgpr("Tmp"), ti.vgpr("Serial")))
        mod.add(ti.VAndB32(ti.vgpr("Tmp"), 63, ti.vgpr("Tmp")))
        mod.add(ti.VLShiftLeftB32(ti.vgpr("Tmp"), 0x2, ti.vgpr("Tmp")))
        mod.addSpaceLine()
        mod.add(ti.DSBPermuteB32(ti.vgpr("OutputB"), ti.vgpr("Tmp"), ti.vgpr("Output")))
        mod.add(ti.SWaitCnt(lgkmcnt=0))
        mod.addSpaceLine()
        mod.add(self.merge_sum())
        mod.add(ti.SLShiftLeftB32(ti.sgpr("Tmp"), 1, ti.sgpr("Tmp")))
        mod.add(ti.SCmpLtU32(ti.sgpr("Tmp"), 64))
        mod.add(ti.SCBranchSCC1(label.getLabelName()))
        mod.addSpaceLine()
        return mod


    def inter_wave_reduction(self) -> ti.Module:
        label_wave_inter = ti.Label("wave_inter", 'wave_inter')
        label_wave_upper = ti.Label("wave_upper", 'wave_upper')
        label_wave_lower = ti.Label("wave_lower", 'wave_lower')
        label_wave_empty = ti.Label("wave_empty", 'wave_empty')
        label_wave_end   = ti.Label("wave_end",   'wave_end')

        mod = ti.Module("inter_wave_reduction")
        mod.addComment0("inter_wave_reduction")
        mod.add(ti.VLShiftRightB32(ti.vgpr("Widx"), int(log2(self.wave_size)), ti.vgpr("Serial")))
        mod.add(ti.SMovB32(ti.sgpr("Offset"), self.num_workitems // self.wave_size))
        mod.add(label_wave_inter)
        mod.add(ti.SLShiftRightB32(ti.sgpr("Offset"), 1, ti.sgpr("Offset")))
        mod.add(ti.SCmpEQU32(ti.sgpr("Offset"), 0))
        mod.add(ti.SCBranchSCC1(label_wave_end.getLabelName()))
        mod.add(ti.SLShiftLeftB32(ti.sgpr("Tmp"), 1, ti.sgpr("Offset")))
        mod.add(ti.VCmpLtU32(ti.sgpr("Tmp+2",2), ti.vgpr("Widx"), ti.sgpr("Tmp")))
        mod.add(ti.VCmpGEU32(ti.sgpr("Tmp+4",2), ti.vgpr("Widx"), ti.sgpr("Offset")))
        mod.add(ti.SAndB64("vcc", ti.sgpr("Tmp+2",2), ti.sgpr("Tmp+4",2)))
        mod.add(ti.SCBranchVCCNZ(label_wave_upper.getLabelName()))
        mod.add(ti.VCmpLtU32("vcc", ti.vgpr("Widx"), ti.sgpr("Offset")))
        mod.add(ti.SCBranchVCCNZ(label_wave_lower.getLabelName()))
        mod.add(ti.SBranch(label_wave_empty.getLabelName()))

        mod.add(label_wave_upper)
        mod.add(ti.VSubU32(ti.vgpr("Tmp"), ti.vgpr("Widx"), ti.sgpr("Offset")))
        mod.add(ti.VLShiftLeftB32(ti.vgpr("Tmp"), int(log2(self.bpe)), ti.vgpr("Tmp")))
        ds = ti.DSModifiers(offset=0)
        DSStorex1 = self.local_write_inst_type(1, self.i_type)
        mod.add(DSStorex1(ti.vgpr("Tmp"), ti.vgpr("Output"), ds))
        mod.add(ti.SWaitCnt(lgkmcnt=0))
        mod.add(ti.SBarrier())
        mod.add(ti.SBranch(label_wave_inter.getLabelName()))
        mod.add(label_wave_lower)
        mod.add(ti.SBarrier())
        mod.add(ti.VLShiftLeftB32(ti.vgpr("Tmp"), int(log2(self.bpe)), ti.vgpr("Widx")))
        ds = ti.DSModifiers(offset=0)
        DSLoadx1 = self.local_read_inst_type(1, self.i_type)
        mod.add(DSLoadx1(ti.vgpr("OutputB"), ti.vgpr("Tmp"), ds))
        mod.add(ti.SWaitCnt(lgkmcnt=0))
        mod.add(self.merge_sum())
        mod.add(ti.SBranch(label_wave_inter.getLabelName()))
        mod.add(label_wave_empty)
        mod.add(ti.SBarrier())
        mod.add(ti.SBranch(label_wave_inter.getLabelName()))
        mod.add(label_wave_end)
        mod.addSpaceLine()
        return mod


    def broadcast(self) -> ti.Module:
        label_lower = ti.Label("broadcast_lower", f'broadcast_lower')
        label_end = ti.Label("broadcast_end", f'broadcast_end')

        mod = ti.Module("broadcast")
        mod.addComment0("broadcast")
        mod.add(ti.VCmpEQU32("vcc", ti.vgpr("Widx"), 0))
        mod.add(ti.SCBranchVCCZ(label_lower.getLabelName()))
        ds = ti.DSModifiers(offset=0)
        DSStorex1 = self.local_write_inst_type(1, self.i_type)
        mod.add(DSStorex1(ti.vgpr("Widx"), ti.vgpr("Output"), ds))
        mod.add(ti.SWaitCnt(lgkmcnt=0))
        mod.add(ti.SBarrier())
        mod.add(ti.SBranch(label_end.getLabelName()))
        mod.add(label_lower)
        mod.add(ti.SBarrier())
        mod.add(ti.VMovB32(ti.vgpr("Tmp"), 0))
        ds = ti.DSModifiers(offset=0)
        DSLoadx1 = self.local_read_inst_type(1, self.i_type)
        mod.add(DSLoadx1(ti.vgpr("Output"), ti.vgpr("Tmp"), ds))
        mod.add(ti.SWaitCnt(lgkmcnt=0))
        mod.add(label_end)
        mod.addSpaceLine()
        mod.addSpaceLine()
        return mod


    def output_result(self) -> ti.Module:
        mod = ti.Module("output_result")
        mod.addComment0("output_result")

        label_break = ti.Label("break", 'break')
        label_break2 = ti.Label("break2", 'break2')
        label_end = ti.Label("end", 'end')
        label_final_loop = ti.Label("final_loop", 'final_loop')
        label_final_output = ti.Label("final_output", 'final_output')
        mod.addSpaceLine()

        mod.add(ti.VReadfirstlaneB32(ti.sgpr("Tmp"), ti.vgpr("Serial")))
        mod.add(ti.SCmpEQU32(ti.sgpr("Tmp"), 0))
        mod.add(ti.SCBranchSCC0(label_end.getLabelName()))
        mod.addSpaceLine()

        if self.arch.find("gfx94") != -1:
            mod.addSpaceLine()
            mod.add(ti.SCmpEQU32(ti.sgpr("NumGroup"), 1))
            mod.add(ti.SCBranchSCC1(label_final_output.getLabelName()))

            mod.add(ti.SLShiftLeftB32(ti.sgpr("Tmp+0"), int(log2(self.i_type.numBytes())), ti.sgpr("NumGroup")))
            mod.add(ti.SMovB32(ti.sgpr("Dst+0"), ti.sgpr("AddressWk+0")))
            mod.add(ti.SMovB32(ti.sgpr("Dst+1"), ti.sgpr("AddressWk+1")))
            mod.add(ti.SMovB32(ti.sgpr("Dst+2"), ti.sgpr("Tmp+0")))
            mod.add(ti.SMovB32(ti.sgpr("Dst+3"), "Srd127_96"))

            mod.add(ti.SLShiftLeftB32(ti.sgpr("Offset"), int(log2(self.i_type.numBytes())), ti.sgpr("WorkGroup0")))
            mod.add(ti.VMovB32(ti.vgpr("Offset"), 0))

            BufferStorex1 = self.global_write_inst_type(1, self.i_type)
            mod.add(BufferStorex1(ti.vgpr("Output"), ti.vgpr("Offset"), ti.sgpr("Dst",4), ti.sgpr("Offset"), ti.MUBUFModifiers(offen=True, glc=True, slc=True)))
            mod.add(ti.SWaitCnt(vmcnt=0))
            mod.addSpaceLine()

            mod.add(ti.SSubI32(ti.sgpr("Tmp"), ti.sgpr("NumGroup"), 1))
            mod.add(ti.SAtomicDec(ti.sgpr("Tmp"), ti.sgpr("AddressSy",2), ti.SMEMModifiers(glc=True)))
            mod.add(ti.SWaitCnt(vmcnt=0, lgkmcnt=0))
            mod.add(ti.SCmpEQU32(ti.sgpr("Tmp"), 1))
            mod.add(ti.SCBranchSCC0(label_end.getLabelName()))
            mod.addSpaceLine()

            mod.add(ti.SLShiftLeftB32(ti.sgpr("Tmp"), int(log2(self.i_type.numBytes())), ti.sgpr("NumGroup")))
            mod.add(ti.SMovB32(ti.sgpr("Src+0"), ti.sgpr("AddressWk+0")))
            mod.add(ti.SMovB32(ti.sgpr("Src+1"), ti.sgpr("AddressWk+1")))
            mod.add(ti.SMovB32(ti.sgpr("Src+2"), ti.sgpr("Tmp")))
            mod.add(ti.SMovB32(ti.sgpr("Src+3"), "Srd127_96"))
            mod.addSpaceLine()

            mod.add(ti.VLShiftLeftB32(ti.vgpr("Offset"), int(log2(self.bpe)), ti.vgpr("Serial")))
            mod.addSpaceLine()

            mod.add(ti.VMovB32(ti.vgpr("Output"), "0"))
            mod.addSpaceLine()

            mod.add(label_final_loop)
            BufferLoadx1 = self.global_read_inst_type(1, self.i_type)
            mod.add(BufferLoadx1(ti.vgpr(f"Value"), ti.vgpr("Offset"), ti.sgpr("Src",4), 0, ti.MUBUFModifiers(offen=True, glc=True, slc=True)))
            mod.add(ti.SWaitCnt(vmcnt=0))
            mod.addSpaceLine()

            mod.add(label_break)

            mod.add(self.max_per_data(0, 1))
            mod.addSpaceLine()

            mod.add(ti.SMovB32(ti.sgpr("Tmp"), self.wave_size * self.i_type.numBytes()))
            mod.add(ti.VAddU32(ti.vgpr("Offset"), ti.vgpr("Offset"), ti.sgpr("Tmp")))
            mod.addSpaceLine()

            mod.add(ti.SSubI32(ti.sgpr("NumGroup"), ti.sgpr("NumGroup"), self.wave_size))
            mod.add(ti.SCmpGtI32(ti.sgpr("NumGroup"), 0))
            mod.add(ti.SCBranchSCC1(label_final_loop.getLabelName()))
            mod.addSpaceLine()

            mod.add(self.intra_wave_reduction("final"))
            mod.addSpaceLine()

            mod.add(label_final_output)

        mod.add(ti.SMovB32(ti.sgpr("Dst+0"), ti.sgpr("AddressOut+0")))
        mod.add(ti.SMovB32(ti.sgpr("Dst+1"), ti.sgpr("AddressOut+1")))
        mod.add(ti.SMovB32(ti.sgpr("Dst+2"), self.o_type.numBytes()))
        mod.add(ti.SMovB32(ti.sgpr("Dst+3"), "Srd127_96"))
        mod.addSpaceLine()

        mod.add(ti.VMovB32(ti.vgpr("Offset"), 0))

        BufferStorex1 = self.global_write_inst_type(1, self.o_type)
        if self.i_type.toChar() == 'H' and self.o_type.toChar() == "S":
            mod.add(ti.VCvtF16toF32(ti.vgpr("Output"), ti.vgpr("Output")))
        elif self.i_type.toChar() == 'S' and self.o_type.toChar() == "H":
            mod.add(ti.VCvtF32toF16(ti.vgpr("Output"), ti.vgpr("Output")))
        mod.add(ti.SNop(1))

        label_divided = ti.Label("divided", 'divided')
        mod.add(ti.SCmpEQI32(ti.sgpr("IsDivided"), 0))
        mod.add(ti.SCBranchSCC1(label_divided.getLabelName()))

        if self.o_type.toChar() == "S":
            mod.add(ti.VRcpF32(ti.vgpr("Output"), ti.vgpr("Output")))
            mod.add(ti.SNop(1))
            mod.add(ti.VMulF32(ti.vgpr("Output"), ti.vgpr("Output"), ti.sgpr("Divided")))
        elif self.o_type.toChar() == "H":
            mod.add(ti.VCvtF32toF16(ti.vgpr("Output"), ti.vgpr("Output")))
            mod.add(ti.SNop(1))
            mod.add(ti.VRcpF32(ti.vgpr("Output"), ti.vgpr("Output")))
            mod.add(ti.SNop(1))
            mod.add(ti.VMulF32(ti.vgpr("Output"), ti.vgpr("Output"), ti.sgpr("Divided")))
            mod.add(ti.SNop(1))
            mod.add(ti.VCvtF16toF32(ti.vgpr("Output"), ti.vgpr("Output")))
            mod.add(ti.SNop(1))
        mod.add(label_divided)

        mod.add(BufferStorex1(ti.vgpr("Output"), ti.vgpr("Offset"), ti.sgpr("Dst",4), 0, ti.MUBUFModifiers(offen=True)))
        mod.addSpaceLine()

        mod.addSpaceLine()
        mod.add(label_end)
        mod.addSpaceLine()
        mod.addSpaceLine()

        return mod


    def amax_kernel_body(self) -> ti.Module:
        mod = ti.Module(self.func_name)
        mod.add(self.defineVariables())
        with asm_func(self.func_name, mod):
            mod.add(self.load_kernel_args())
            mod.add(self.init_param())
            mod.add(self.calculate_global_address())
            mod.add(self.sum_per_blocksize())
            mod.add(self.adjust_global_address())
            mod.add(self.sum_per_threadx4x4())
            mod.add(self.sum_per_threadx4())
            mod.add(self.adjust_global_address_2())
            mod.add(self.sum_per_thread())
            mod.add(self.sum_in_some_thread())
            mod.add(self.label_reduction)
            mod.add(self.intra_wave_reduction("middle"))
            mod.add(self.inter_wave_reduction())
            mod.add(self.broadcast())
            mod.add(self.output_result())
        return mod

@dataclass
class KernelArgument:
    size: int
    offset: int
    value_kind: str
    address_space: Optional[str] = None

    def to_dict(self):
        d = {'.size': self.size, '.offset': self.offset,
             '.value_kind': self.value_kind}

        if self.address_space:
            d['.address_space'] = self.address_space

        return d

@dataclass
class KernelMeta:
    name: str
    num_vgpr: int
    num_sgpr: int
    num_agpr: int
    num_lds_bytes: int
    wavefront_size: int
    max_workgroup_size: int
    args_alignment: int
    args: List[KernelArgument]

    def update_args_offsets(self):
        offset = 0
        for arg in args:
            arg.offset = offset
            offset += arg.size

    def _get_args_size(self):
        total_size = sum(arg.size for arg in self.args)
        total_size += (self.args_alignment - (total_size % self.args_alignment))
        return total_size

    def to_dict(self):
        return {
            '.name': self.name,
            '.symbol': f'{self.name}.kd',
            '.kernarg_segment_size': self._get_args_size(),
            '.group_segment_fixed_size': self.num_lds_bytes,
            '.private_segment_fixed_size': 0,
            '.kernarg_segment_align': self.args_alignment,
            '.wavefront_size': self.wavefront_size,
            '.sgpr_count': self.num_sgpr,
            '.vgpr_count': self.num_vgpr,
            '.agpr_count': self.num_agpr,
            '.max_flat_workgroup_size': self.max_workgroup_size,
            '.args': [arg.to_dict() for arg in self.args]
        }

def meta_str(kernels: Tuple[KernelMeta]):
    beg = '.amdgpu_metadata\n---'
    content_str = yaml.dump({'amdhsa.version': [1, 1], 'amdhsa.kernels': [kernel.to_dict() for kernel in kernels]})
    end = '.end_amdgpu_metadata'
    return '\n'.join([beg, content_str, end, ''])


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-o', '--output', type=str, required=True, help='Output path of compiled binary')
    ap.add_argument('-t', type=str, default="H", help='data type')
    ap.add_argument('-d', type=str, default="S", help='dest data type')
    ap.add_argument('-s', type=str, default="F8", help='scale data type')
    ap.add_argument('--toolchain', type=str, default='/opt/rocm/llvm/bin/clang++', help='Path to ROCm compiler')
    ap.add_argument('--debug-build', action='store_true', dest='debug_build', help='Build with debug information')
    ap.add_argument('--is-scale', action='store_true', dest='is_scale', help='Enable scaled output or not')
    ap.add_argument('--arch', type=str, default='gfx942', help='Target architecture for assembler, e.g. gfx908. Default is gfx90a')
    ap.set_defaults(debug_build=False)

    args = ap.parse_args()
    output_path: str = args.output
    t: str = args.t
    d: str = t if (args.d =="None") else args.d
    s: str = args.s
    num_workitem: int = 256
    wave_size: int = 64
    c: int = 8
    toolchain_path: str = args.toolchain
    debug_build: bool = args.debug_build
    arch: str = args.arch
    is_scale: bool = args.is_scale
    isa = gfxArch(arch)

    if any([not i for i in (arch, toolchain_path, isa)]):
        restoreDefaultGlobalParameters()
        assignGlobalParameters({})
        detectGlobalCurrentISA()
        isa = globalParameters['CurrentISA']
        arch = getGfxName(isa)
        toolchain_path = globalParameters['AssemblerPath']

    ti.Base._global_ti.init(isa, toolchain_path, False)
    amax = AMaxKernelGenerator(ti.DataType(t), ti.DataType(d), ti.DataType(s), num_workitem, wave_size, c, int(4 / ti.DataType(t).numRegisters()), arch, is_scale)
    kernel_body = amax.amax_kernel_body()
    args = amax.kernel_args()
    func_name = amax.func_name
    meta = KernelMeta(func_name, amax.vgpr_pool.size(), amax.sgpr_pool.size(), 0, amax.lds_usage_byte, wave_size, num_workitem, 8, args)
    meta.update_args_offsets()
    k_str = '\n'.join([kernel_header(func_name, arch, amax.vgpr_pool.size(), amax.sgpr_pool.size(), amax.lds_usage_byte),
                       meta_str((meta,)),
                       str(kernel_body)])

    with open(output_path, 'w') as f:
        f.write(k_str)

    output_path_basename = os.path.splitext(output_path)[0]

    if debug_build:
        build_args = ['-x', 'assembler', '-target', 'amdgcn-amd-amdhsa', '-mcode-object-version=4', f'-mcpu={arch}', '-mwavefrontsize64', '-c', '-g', '-o', f'{output_path_basename}.o', f'{output_path_basename}.s']
    else:
        build_args = ['-x', 'assembler', '-target', 'amdgcn-amd-amdhsa', '-mcode-object-version=4', f'-mcpu={arch}', '-mwavefrontsize64', '-c', '-o', f'{output_path_basename}.o', f'{output_path_basename}.s']

    ret = subprocess.run([toolchain_path] + build_args)
    ret = subprocess.run([toolchain_path, '-target', 'amdcgn-amdhsa', '-o', f'{output_path_basename}.co', f'{output_path_basename}.o'])
    amax.dump('yaml', f'{output_path_basename}.yaml')

