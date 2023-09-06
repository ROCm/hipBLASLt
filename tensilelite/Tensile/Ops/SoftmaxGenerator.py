################################################################################
#
# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
from contextlib import contextmanager
import Tensile.TensileInstructions as ti
from Tensile.Common import detectGlobalCurrentISA, restoreDefaultGlobalParameters, \
    assignGlobalParameters, getGfxName, gfxArch, globalParameters

def record_num_calls(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not hasattr(wrapper, '__num_calls__'):
            wrapper.__num_calls__ = 0
            wrapper.num_calls = lambda: wrapper.__num_calls__
        wrapper.__num_calls__ += 1
        return f(*args, **kwargs)

    return wrapper

def kernel_header(name: str, gfx_arch: str):
    return f'''.amdgcn_target "amdgcn-amd-amdhsa--{gfx_arch}"
.text
.global {name}
.p2align 8
.type {name},@function
            '''

@contextmanager
def asm_func(func_name: str, module: ti.Module):
    try:
        module.add(ti.TextBlock(f'{func_name}:\n'))
        yield
    finally:
        end_label_name = f'.L{func_name}_end'
        module.add(ti.TextBlock(f'{end_label_name}:\n'))
        module.add(ti.TextBlock(f'.size {func_name}, {end_label_name} - {func_name}\n'))

@contextmanager
def auto_exec_scope(sgpr_pool: ti.RegisterPool, module: ti.Module):
    try:
        tmp_exec_reg_idx = sgpr_pool.checkOutAligned(2, 2)
        module.add(ti.SMovB64(ti.sgpr(tmp_exec_reg_idx, 2), ti.EXEC()))
        yield
    finally:
        module.add(ti.SMovB64(ti.EXEC(), ti.sgpr(tmp_exec_reg_idx, 2)))
        sgpr_pool.checkIn(tmp_exec_reg_idx)

class SoftmaxKernelGenerator:
    srd_num_reg = 4
    srd_alignment = 4

    def __init__(self,
                 io_type: ti.DataType,
                 num_cols: int,
                 num_rows: int,
                 num_workitems: int,
                 arch: str):
        self.io_type = io_type
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.num_workitems = num_workitems
        self.sgpr_pool = ti.RegisterPool(18, 's', True)
        self.vgpr_pool = ti.RegisterPool(10, 'v', True)
        self.sgpr_pool.addRange(3, 17) #TODO: estimate this
        self.vgpr_pool.addRange(1, 9) #TODO: estimate this
        self.t_id_reg_idx = 0 #TODO: support config on this
        self.wg_id_reg_idx = 2 #TODO: support config on this
        self.numerically_stable = True
        self.debug_label = True
        self.arch = arch
        self.op = 'Softmax'

    def _validate(self):
        assert self.num_cols * self.num_rows == self.num_workitems

    @property
    def lds_usage_byte(self) -> int:
        return self.num_cols * self.num_rows * self.io_type.numBytes()

    @property
    def func_name(self):
        return f'Softmax_DT_{self.io_type}_MT_{self.num_rows}_{self.num_cols}'

    def dumps(self, format: str) -> str:
        param_dict = {
            'io_type': self.io_type.toChar(),
            'num_cols': self.num_cols,
            'num_rows': self.num_rows,
            'num_workitems': self.num_workitems,
            'func_name': self.func_name,
            'numerically_stable': self.numerically_stable,
            'debug_label': self.debug_label,
            'arch': self.arch,
            'op': self.op
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

    def loads(self, format: str, data: str):
        if format.lower() == 'yaml':
            param_dict = yaml.load(data, yaml.SafeLoader)
        elif format.lower() == 'json':
            param_dict = json.loads(data)
        else:
            assert False, f'Unsupported format: {format}'

        self.io_type = ti.DataType(param_dict['io_type'])
        self.num_cols = param_dict['num_cols']
        self.num_rows = param_dict['num_rows']
        self.num_workitems = param_dict['num_workitems']
        self.func_name = param_dict['func_name']
        self.numerically_stable = param_dict['numerically_stable']
        self.debug_label = param_dict['debug_label']
        self.arch = param_dict['arch']
        self.op = param_dict['op']

    def local_write_inst_type(self, num_elements: int):
        if self.io_type.isSingle():
            insts = {
                1: ti.DSStoreB32,
                2: ti.DSStoreB64,
                4: ti.DSStoreB128
            }

        return insts[num_elements]

    def global_read_inst_type(self, num_elements: int) -> Union[ti.BufferLoadB32, ti.BufferLoadB64, ti.BufferLoadB128]:
        if self.io_type.isSingle():
            insts = {
                1: ti.BufferLoadB32,
                2: ti.BufferLoadB64,
                4: ti.BufferLoadB128
            }
        elif self.io_type.isHalf():
            insts = {
                2: ti.BufferLoadB32,
                4: ti.BufferLoadB64,
                8: ti.BufferLoadB128
            }
        else:
            raise NotImplementedError
        return insts[num_elements]

    def global_write_inst_type(self, num_elements: int):
        if self.io_type.isSingle():
            insts = {
                1: ti.BufferStoreB32,
                2: ti.BufferStoreB64,
                4: ti.BufferStoreB128
            }
        elif self.io_type.isHalf():
            insts = {
                2: ti.BufferStoreB32,
                4: ti.BufferStoreB64,
                8: ti.BufferStoreB128
            }
        else:
            raise NotImplementedError
        return insts[num_elements]

    @property
    def srd_const(self) -> str:
        if self.io_type.isSingle():
            return hex(0x20000)

        raise NotImplementedError

    @property
    def bpe(self) -> int:
        return self.io_type.numBytes()

    def load_kernel_args(self):
        kernel_args_addr = 0
        kernel_args_addr_size = 2
        input_srd_idx = self.sgpr_pool.checkOutAligned(self.srd_num_reg, self.srd_alignment)
        output_srd_idx = self.sgpr_pool.checkOutAligned(self.srd_num_reg, self.srd_alignment)
        m_reg_idx = self.sgpr_pool.checkOut(1)
        n_reg_idx = self.sgpr_pool.checkOut(1)
        num_elem_reg_idx = self.sgpr_pool.checkOut(1)
        module = ti.Module('Load kernel args')
        module.add(ti.SLoadB64(ti.sgpr(input_srd_idx, 2), ti.sgpr(kernel_args_addr, kernel_args_addr_size), 0))
        module.add(ti.SLoadB32(ti.sgpr(m_reg_idx), ti.sgpr(kernel_args_addr, kernel_args_addr_size), 16))
        module.add(ti.SLoadB32(ti.sgpr(n_reg_idx), ti.sgpr(kernel_args_addr, kernel_args_addr_size), 20))
        module.add(ti.SWaitCnt(lgkmcnt=0))
        module.add(ti.SLoadB64(ti.sgpr(output_srd_idx, 2), ti.sgpr(kernel_args_addr, kernel_args_addr_size), 8))
        module.add(ti.SMulI32(ti.sgpr(num_elem_reg_idx), ti.sgpr(m_reg_idx), ti.sgpr(n_reg_idx)))
        module.add(ti.SMulI32(ti.sgpr(num_elem_reg_idx), ti.sgpr(num_elem_reg_idx), hex(self.bpe)))
        module.add(ti.SMovB32(ti.sgpr(input_srd_idx + 2), ti.sgpr(num_elem_reg_idx)))
        module.add(ti.SWaitCnt(lgkmcnt=0))
        module.add(ti.SMovB32(ti.sgpr(output_srd_idx + 2), ti.sgpr(num_elem_reg_idx)))
        module.add(ti.SMovB32(ti.sgpr(input_srd_idx + 3), self.srd_const))
        module.add(ti.SMovB32(ti.sgpr(output_srd_idx + 3), self.srd_const))
        self.sgpr_pool.checkIn(num_elem_reg_idx)
        return module, input_srd_idx, output_srd_idx, m_reg_idx, n_reg_idx

    def local_offset(self, stride_reg_idx: Optional[int]) -> Tuple[ti.Module, int]:
        module = ti.Module()
        t_id_reg_idx = self.t_id_reg_idx
        tmp_reg_idx = self.vgpr_pool.checkOut(2)
        col_idx_reg_idx = tmp_reg_idx
        row_idx_reg_idx = tmp_reg_idx + 1
        byte_offset_reg_idx = self.vgpr_pool.checkOut(1)
        module.add(ti.vectorStaticDivideAndRemainder(row_idx_reg_idx, col_idx_reg_idx, t_id_reg_idx, self.num_cols, None))

        if not stride_reg_idx:
            module.add(ti.staticMultiply(ti.vgpr(byte_offset_reg_idx), ti.vgpr(row_idx_reg_idx), self.num_cols, None))
        else:
            module.add(ti.VMulLOU32(ti.vgpr(byte_offset_reg_idx), ti.vgpr(row_idx_reg_idx), ti.sgpr(stride_reg_idx)))

        module.add(ti.VAddU32(ti.vgpr(byte_offset_reg_idx), ti.vgpr(byte_offset_reg_idx), ti.vgpr(col_idx_reg_idx)))
        module.add(ti.staticMultiply(ti.vgpr(byte_offset_reg_idx), ti.vgpr(byte_offset_reg_idx), self.bpe, None))
        self.vgpr_pool.checkIn(tmp_reg_idx)
        return module, byte_offset_reg_idx

    @record_num_calls
    def global_read(self, srd_reg_idx: Union[int, str],
                    soffset_reg_idx: int,
                    n_reg_idx: int, sync: bool):
        '''
        col_idx = t_id % num_cols
        row_idx = t_id / num_cols
        byte_offset = row_idx * num_cols + col_idx
        byte_offset *= bpe
        data = global_read(src, byte_offset)
        '''
        module = ti.Module('global read')

        if self.debug_label:
            module.add(ti.Label(f'global_read_{self.global_read.num_calls()}', 'global read begins'))

        local_offset_mod, byte_offset_reg_idx = self.local_offset(n_reg_idx)
        module.add(local_offset_mod)

        num_elem_read = 1
        BufferLoadType = self.global_read_inst_type(num_elem_read)
        data_reg_idx = self.vgpr_pool.checkOut(1)
        module.add(BufferLoadType(ti.vgpr(data_reg_idx), ti.vgpr(byte_offset_reg_idx), ti.sgpr(srd_reg_idx, self.srd_num_reg), ti.sgpr(soffset_reg_idx), ti.MUBUFModifiers(offen=True)))
        self.vgpr_pool.checkIn(byte_offset_reg_idx)

        if sync:
            module.add(ti.SWaitCnt(vmcnt=0))

        return module, data_reg_idx
        
    def local_read(self, ext_local_byte_offset_reg_idx: Optional[int] = None, sync: bool = True):
        module = ti.Module()

        if not ext_local_byte_offset_reg_idx:
            local_offset_mod, local_byte_offset_reg_idx = self.local_offset()
            module.add(local_offset_mod)
        else:
            local_byte_offset_reg_idx = ext_local_byte_offset_reg_idx

        data_reg_idx = self.vgpr_pool.checkOut(1)
        module.add(ti.DSLoadB32(ti.vgpr(data_reg_idx), ti.vgpr(local_byte_offset_reg_idx), False))

        if sync:
            module.add(ti.SWaitCnt(lgkmcnt=0))

        if not ext_local_byte_offset_reg_idx:
            self.vgpr_pool.checkIn(local_byte_offset_reg_idx)

        return module, data_reg_idx

    def setup_global_read_wg_offset(self, stride0_reg_idx: int) -> Tuple[ti.Module, int]:
        '''
        wg_id = 2
        num_row_proc = num_workitems / num_cols
        byte_offset = num_row_proc * ld
        byte_offset *= bpe
        '''
        mod = ti.Module()

        if self.debug_label:
            mod.add(ti.Label('setup_global_read_wg_offset', 'setup global read wg offset begins'))

        wg_id_reg_idx = self.wg_id_reg_idx
        wg_byte_offset_reg_idx = self.sgpr_pool.checkOut(1)
        byte_offset = (self.num_workitems // self.num_cols) * self.bpe
        mod.add(ti.SMulI32(ti.sgpr(wg_byte_offset_reg_idx), ti.sgpr(wg_id_reg_idx), hex(byte_offset)))
        mod.add(ti.SMulI32(ti.sgpr(wg_byte_offset_reg_idx), ti.sgpr(wg_byte_offset_reg_idx), ti.sgpr(stride0_reg_idx)))
        return mod, wg_byte_offset_reg_idx

    @record_num_calls
    def local_write(self, data_reg_idx: int, num_elem: int, ext_local_read_byte_offset_reg_idx: Optional[int], sync: bool):
        module = ti.Module()

        if self.debug_label:
            module.add(ti.Label(f'local_write_{self.local_write.num_calls()}', 'local write begins'))

        if not ext_local_read_byte_offset_reg_idx:
            local_offset_mod, byte_offset_reg_idx = self.local_offset()
            module.add(local_offset_mod)
        else:
            byte_offset_reg_idx = ext_local_read_byte_offset_reg_idx

        LocalWriteType = self.local_write_inst_type(num_elem)
        module.add(LocalWriteType(ti.vgpr(byte_offset_reg_idx), ti.vgpr(data_reg_idx)))

        if not ext_local_read_byte_offset_reg_idx:
            self.vgpr_pool.checkIn(byte_offset_reg_idx)

        if sync:
            module.add(ti.SWaitCnt(lgkmcnt=0))
            module.add(ti.SBarrier())

        return module

    def lds_max(self, lds_addr0: int, lds_addr1: int, sync: bool = True) -> ti.Module:
        '''
        lds[lds_addr0] = max(lds[lds_addr0], lds[lds_addr1])
        '''
        module = ti.Module()
        data_reg_idx_0 = self.vgpr_pool.checkOut(2)
        data_reg_idx_1 = data_reg_idx_0 + 1
        max_reg_idx = data_reg_idx_0
        module.add(ti.DSLoadB32(ti.vgpr(data_reg_idx_0), ti.vgpr(lds_addr0), False))
        module.add(ti.DSLoadB32(ti.vgpr(data_reg_idx_1), ti.vgpr(lds_addr1), False))
        module.add(ti.SWaitCnt(lgkmcnt=0))
        module.add(ti.VMaxF32(ti.vgpr(max_reg_idx), ti.vgpr(data_reg_idx_0), ti.vgpr(data_reg_idx_1)))
        module.add(ti.DSStoreB32(ti.vgpr(lds_addr0), ti.vgpr(max_reg_idx)))

        if sync:
            module.add(ti.SWaitCnt(lgkmcnt=0))
            module.add(ti.SBarrier())

        self.vgpr_pool.checkIn(data_reg_idx_0)
        return module

    @record_num_calls
    def max_elem(self) -> Tuple[ti.Module, int]:
        mod = ti.Module()

        if self.debug_label:
            mod.add(ti.Label(f'get_row_head_{self.max_elem.num_calls()}', 'get row head begins'))

        t_id_reg_idx = self.t_id_reg_idx
        addr_reg_idx = self.vgpr_pool.checkOut(1)
        mod.add(ti.vectorStaticDivide(addr_reg_idx, t_id_reg_idx, self.num_cols, None))
        mod.add(ti.staticMultiply(ti.vgpr(addr_reg_idx), ti.vgpr(addr_reg_idx), self.bpe * self.num_cols, None))
        mod.add(ti.DSLoadB32(ti.vgpr(addr_reg_idx), ti.vgpr(addr_reg_idx), False))
        mod.add(ti.SWaitCnt(lgkmcnt=0))
        return mod, addr_reg_idx

    def sum_elem(self) -> Tuple[ti.Module, int]:
        return self.max_elem()

    def lds_sum(self, lds_addr0: int, lds_addr1: int, sync: bool = True):
        '''
        lds[lds_addr0] += lds[lds_addr1]
        '''
        module = ti.Module()
        data_reg_idx_0 = self.vgpr_pool.checkOut(2)
        data_reg_idx_1 = data_reg_idx_0 + 1
        sum_reg_idx = data_reg_idx_0
        module.add(ti.DSLoadB32(ti.vgpr(data_reg_idx_0), ti.vgpr(lds_addr0), False))
        module.add(ti.DSLoadB32(ti.vgpr(data_reg_idx_1), ti.vgpr(lds_addr1), False))
        module.add(ti.SWaitCnt(lgkmcnt=0))
        module.add(ti.VAddF32(ti.vgpr(sum_reg_idx), ti.vgpr(data_reg_idx_0), ti.vgpr(data_reg_idx_1)))
        module.add(ti.DSStoreB32(ti.vgpr(lds_addr0), ti.vgpr(sum_reg_idx)))

        if sync:
            module.add(ti.SWaitCnt(lgkmcnt=0))
            module.add(ti.SBarrier())

        self.vgpr_pool.checkIn(data_reg_idx_0)
        return module

    def reduction_sum(self, n_reg_idx: int) -> ti.Module:
        module = ti.Module()

        if self.debug_label:
            module.add(ti.Label('reduction_sum', 'reduction max begins'))

        num_iter = round(log2(self.num_cols))
        l_reg_idx = self.vgpr_pool.checkOut(4)
        r_reg_idx = l_reg_idx + 1
        col_reg_idx = l_reg_idx + 2
        byte_offset_reg_idx = l_reg_idx + 3
        t_id_reg_idx = self.t_id_reg_idx
        module.add(ti.vectorStaticDivideAndRemainder(byte_offset_reg_idx, col_reg_idx, t_id_reg_idx, self.num_cols, None))
        module.add(ti.staticMultiply(ti.vgpr(byte_offset_reg_idx), ti.vgpr(byte_offset_reg_idx), self.num_cols, None))
        module.add(ti.VAddU32(ti.vgpr(byte_offset_reg_idx), ti.vgpr(byte_offset_reg_idx), ti.vgpr(col_reg_idx)))
        module.add(ti.staticMultiply(ti.vgpr(byte_offset_reg_idx), ti.vgpr(byte_offset_reg_idx), self.bpe, None))
        module.add(ti.VMovB32(ti.vgpr(l_reg_idx), ti.vgpr(byte_offset_reg_idx)))

        with ti.allocTmpGpr(self.sgpr_pool, 2, self.sgpr_pool.size(), 1) as tmp_sgpr_res:
            cmp_res_reg_idx = tmp_sgpr_res.idx
            reduction_col_reg_idx = self.vgpr_pool.checkOut(1)

            for i in range(num_iter):
                with auto_exec_scope(self.sgpr_pool, module):
                    s = self.num_cols >> (i + 1)
                    assert s > 0
                    cmp_byte_rel_offset = s * self.bpe
                    module.add(ti.VAddU32(ti.vgpr(reduction_col_reg_idx), hex(s), ti.vgpr(col_reg_idx)))
                    module.add(ti.VCmpLeU32(ti.sgpr(cmp_res_reg_idx, 2), ti.vgpr(col_reg_idx), ti.sgpr(n_reg_idx)))
                    module.add(ti.VCmpGtU32(ti.VCC(), hex(s), ti.vgpr(col_reg_idx)))
                    module.add(ti.SAndB64(ti.sgpr(cmp_res_reg_idx, 2), ti.sgpr(cmp_res_reg_idx, 2), ti.VCC()))
                    module.add(ti.VCmpLtU32(ti.VCC(), ti.vgpr(reduction_col_reg_idx), ti.sgpr(n_reg_idx)))
                    module.add(ti.SAndB64(ti.EXEC(), ti.sgpr(cmp_res_reg_idx, 2), ti.VCC()))
                    module.add(ti.VAddU32(ti.vgpr(r_reg_idx), hex(cmp_byte_rel_offset), ti.vgpr(l_reg_idx)))
                    module.add(self.lds_sum(l_reg_idx, r_reg_idx))

            self.vgpr_pool.checkIn(reduction_col_reg_idx)

        self.vgpr_pool.checkIn(l_reg_idx)
        return module

    def reduction_max(self, n_reg_idx: Union[int, str]) -> ti.Module:
        module = ti.Module()

        if self.debug_label:
            module.add(ti.Label('reduction_max', 'reduction max begins'))

        num_iter = round(log2(self.num_cols))
        l_reg_idx = self.vgpr_pool.checkOut(4)
        r_reg_idx = l_reg_idx + 1
        col_reg_idx = l_reg_idx + 2
        byte_offset_reg_idx = l_reg_idx + 3
        t_id_reg_idx = self.t_id_reg_idx
        module.add(ti.vectorStaticDivideAndRemainder(byte_offset_reg_idx, col_reg_idx, t_id_reg_idx, self.num_cols, None))
        module.add(ti.staticMultiply(ti.vgpr(byte_offset_reg_idx), ti.vgpr(byte_offset_reg_idx), self.num_cols, None))
        module.add(ti.VAddU32(ti.vgpr(byte_offset_reg_idx), ti.vgpr(byte_offset_reg_idx), ti.vgpr(col_reg_idx)))
        module.add(ti.staticMultiply(ti.vgpr(byte_offset_reg_idx), ti.vgpr(byte_offset_reg_idx), self.bpe, None))
        module.add(ti.VMovB32(ti.vgpr(l_reg_idx), ti.vgpr(byte_offset_reg_idx)))

        with ti.allocTmpGpr(self.sgpr_pool, 2, self.sgpr_pool.size(), 1) as tmp_sgpr_res:
            cmp_res_reg_idx = tmp_sgpr_res.idx
            reduction_col_reg_idx = self.vgpr_pool.checkOut(1)

            for i in range(num_iter):
                with auto_exec_scope(self.sgpr_pool, module):
                    s = self.num_cols >> (i + 1)
                    assert s > 0
                    cmp_byte_rel_offset = s * self.bpe
                    module.add(ti.VAddU32(ti.vgpr(reduction_col_reg_idx), hex(s), ti.vgpr(col_reg_idx)))
                    module.add(ti.VCmpLeU32(ti.sgpr(cmp_res_reg_idx, 2), ti.vgpr(col_reg_idx), ti.sgpr(n_reg_idx)))
                    module.add(ti.VCmpGtU32(ti.VCC(), hex(s), ti.vgpr(col_reg_idx)))
                    module.add(ti.SAndB64(ti.sgpr(cmp_res_reg_idx, 2), ti.sgpr(cmp_res_reg_idx, 2), ti.VCC()))
                    module.add(ti.VCmpLtU32(ti.VCC(), ti.vgpr(reduction_col_reg_idx), ti.sgpr(n_reg_idx)))
                    module.add(ti.SAndB64(ti.EXEC(), ti.sgpr(cmp_res_reg_idx, 2), ti.VCC()))
                    module.add(ti.VAddU32(ti.vgpr(r_reg_idx), hex(cmp_byte_rel_offset), ti.vgpr(l_reg_idx)))
                    module.add(self.lds_max(l_reg_idx, r_reg_idx))

            self.vgpr_pool.checkIn(reduction_col_reg_idx)

        self.vgpr_pool.checkIn(l_reg_idx)
        return module

    def sub_max(self, data_reg_idx, max_elem_reg_idx) -> ti.Module:
        mod = ti.Module()
        mod.add(ti.VSubF32(ti.vgpr(data_reg_idx), ti.vgpr(data_reg_idx), ti.vgpr(max_elem_reg_idx)))
        return mod

    def exp(self, data_reg_idx) -> ti.Module:
        mod = ti.Module()

        if self.debug_label:
            mod.add(ti.Label('exp', 'exp begins'))

        mod.add(ti.VMulF32(ti.vgpr(data_reg_idx), 1.0 / log(2), ti.vgpr(data_reg_idx)))
        mod.add(ti.VExpF32(ti.vgpr(data_reg_idx), ti.vgpr(data_reg_idx)))

        if ti.Base._global_ti.getArchCaps()["TransOpWait"]:
            mod.add(ti.SNop(waitState=0, comment="1 wait states"))

        return mod

    def div_sum(self, data_reg_idx, sum_reg_idx, safe: bool = False) -> ti.Module:
        mod = ti.Module()

        if self.debug_label:
            mod.add(ti.Label('div_sum', 'div sum begins'))

        assert safe is False, 'currently support plain div only'

        mod.add(ti.VRcpF32(ti.vgpr(sum_reg_idx), ti.vgpr(sum_reg_idx)))

        if ti.Base._global_ti.getArchCaps()["TransOpWait"]:
            mod.add(ti.SNop(waitState=0, comment="1 wait states"))

        mod.add(ti.VMulF32(ti.vgpr(data_reg_idx), ti.vgpr(data_reg_idx), ti.vgpr(sum_reg_idx)))
        return mod

    def global_write_data(self,
                          data_reg_idx: int,
                          srd_reg_idx: int,
                          n_reg_idx: int,
                          wg_byte_offset_reg_idx: int,
                          sync: bool):
        module = ti.Module()

        if self.debug_label:
            module.add(ti.Label('global_write_data', 'global write begins'))

        with auto_exec_scope(self.sgpr_pool, module):
            col_reg_idx = self.vgpr_pool.checkOut(1)
            module.add(ti.vectorStaticRemainder(None, col_reg_idx, self.t_id_reg_idx, self.num_cols, None, None))
            module.add(ti.VCmpXLtU32(ti.VCC(), ti.vgpr(col_reg_idx), ti.sgpr(n_reg_idx)))
            self.vgpr_pool.checkIn(col_reg_idx)
            local_offset_mod, local_byte_offset_reg_idx = self.local_offset(n_reg_idx)
            module.add(local_offset_mod)

            GlobalWriteInstType = self.global_write_inst_type(1)
            module.add(GlobalWriteInstType(ti.vgpr(data_reg_idx), ti.vgpr(local_byte_offset_reg_idx), ti.sgpr(srd_reg_idx, self.srd_num_reg), ti.sgpr(wg_byte_offset_reg_idx), ti.MUBUFModifiers(offen=True)))

            if sync:
                module.add(ti.SWaitCnt(vmcnt=0))

            self.vgpr_pool.checkIn(local_byte_offset_reg_idx)
        return module

    def kernel_args(self):
        return (KernelArgument(8, 0, 'global_buffer', 'global'),
                KernelArgument(8, 8, 'global_buffer', 'global'),
                KernelArgument(4, 16, 'by_value'),
                KernelArgument(4, 20, 'by_value'))

    def softmax_kernel_body(self):
        mod = ti.Module(self.func_name)
        with asm_func(self.func_name, mod):
            kernel_args_load_mod, input_srd, output_srd, m_reg_idx, n_reg_idx = self.load_kernel_args()
            mod.add(kernel_args_load_mod)
            wg_offset_mod, wg_offset_reg_idx = self.setup_global_read_wg_offset(n_reg_idx)
            mod.add(wg_offset_mod)
            local_offset_mod, local_offset_byte_offset_reg_idx = self.local_offset(None)
            mod.add(local_offset_mod)
            gl_mod, data_reg_idx = self.global_read(input_srd, wg_offset_reg_idx, n_reg_idx, True)
            mod.add(gl_mod)
            mod.add(self.local_write(data_reg_idx, 1, local_offset_byte_offset_reg_idx, True))

            if self.numerically_stable:
                mod.add(self.reduction_max(n_reg_idx))
                max_addr_mod, max_reg_idx = self.max_elem()
                mod.add(max_addr_mod)
                mod.add(self.sub_max(data_reg_idx, max_reg_idx))
                self.vgpr_pool.checkIn(max_reg_idx)

            mod.add(self.exp(data_reg_idx))
            mod.add(self.local_write(data_reg_idx, 1, local_offset_byte_offset_reg_idx, True))
            mod.add(self.reduction_sum(n_reg_idx))
            sum_elem_mod, sum_reg_idx = self.sum_elem()
            mod.add(sum_elem_mod)
            mod.add(self.div_sum(data_reg_idx, sum_reg_idx))
            self.vgpr_pool.checkIn(sum_reg_idx)
            gw_mod = self.global_write_data(data_reg_idx, output_srd, n_reg_idx, wg_offset_reg_idx, True)
            mod.add(gw_mod)
            mod.add(ti.SEndpgm())
            self.sgpr_pool.checkIn(input_srd)
            self.sgpr_pool.checkIn(output_srd)
            self.sgpr_pool.checkIn(wg_offset_reg_idx)
            self.sgpr_pool.checkIn(m_reg_idx)
            self.sgpr_pool.checkIn(n_reg_idx)
            self.vgpr_pool.checkIn(data_reg_idx)
            self.vgpr_pool.checkIn(local_offset_byte_offset_reg_idx)
        return mod

def kernel_rodata(name: str):
    return f'''
.rodata
.p2align 6
.amdhsa_kernel {name}
.amdhsa_user_sgpr_kernarg_segment_ptr 1
.amdhsa_system_sgpr_workgroup_id_x 1
.amdhsa_accum_offset 8
.amdhsa_next_free_vgpr .amdgcn.next_free_vgpr
.amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
.end_amdhsa_kernel
'''

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
    return '\n'.join([beg, content_str, end])


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-o', '--output', type=str, required=True, help='Output path of compiled binary')
    ap.add_argument('-m', type=int, default=16, help='Dimension 0 of tile')
    ap.add_argument('-n', type=int, default=16, help='Dimension 1 of tile')
    ap.add_argument('--toolchain', type=str, default='/opt/rocm/llvm/bin/clang++', help='Path to ROCm compiler')
    ap.add_argument('--debug-build', action='store_true', dest='debug_build', help='Build with debug information')
    ap.set_defaults(debug_build=False)
    ap.add_argument('--arch', type=str, default='gfx90a', help='Target architecture for assembler, e.g. gfx908. Default is gfx90a')
    args = ap.parse_args()
    output_path: str = args.output
    m: int = args.m
    n: int = args.n
    toolchain_path: str = args.toolchain
    debug_build: bool = args.debug_build
    arch: str = args.arch
    isa = gfxArch(arch)

    if any([not i for i in (arch, toolchain_path, isa)]):
        restoreDefaultGlobalParameters()
        assignGlobalParameters({})
        detectGlobalCurrentISA()
        isa = globalParameters['CurrentISA']
        arch = getGfxName(isa)
        toolchain_path = globalParameters['AssemblerPath']

    ti.Base._global_ti.init(isa, toolchain_path, False)
    softmax = SoftmaxKernelGenerator(ti.DataType('S'), n, m, 256, arch)
    kernel_body = softmax.softmax_kernel_body()
    args = softmax.kernel_args()
    func_name = softmax.func_name
    meta = KernelMeta(func_name, softmax.vgpr_pool.size(), softmax.sgpr_pool.size(), 0, softmax.lds_usage_byte, 64, 256, 8, args)
    meta.update_args_offsets()
    k_str = '\n'.join([kernel_header(func_name, arch),
                       str(kernel_body),
                       kernel_rodata(func_name),
                       meta_str((meta,))])

    with open(output_path, 'w') as f:
        f.write(k_str)

    output_path_basename = os.path.splitext(output_path)[0]

    if debug_build:
        build_args = ['-x', 'assembler', '-target', 'amdgcn-amd-amdhsa', '-mcode-object-version=4', f'-mcpu={arch}', '-mwavefrontsize64', '-c', '-g', '-o', f'{output_path_basename}.o', f'{output_path_basename}.s']
    else:
        build_args = ['-x', 'assembler', '-target', 'amdgcn-amd-amdhsa', '-mcode-object-version=4', f'-mcpu={arch}', '-mwavefrontsize64', '-c', '-o', f'{output_path_basename}.o', f'{output_path_basename}.s']

    ret = subprocess.run([toolchain_path] + build_args)
    ret = subprocess.run([toolchain_path, '-target', 'amdcgn-amdhsa', '-o', f'{output_path_basename}.co', f'{output_path_basename}.o'])
    softmax.dump('yaml', f'{output_path_basename}.yaml')
