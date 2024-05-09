################################################################################
#
# Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gemmmodel'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gemmmodel/gmodel_lib'))

from gmodel_lib.problem import GemmProblemFromSizes
from gmodel_lib.gemm_model import GemmModel
from gmodel_lib.gemm_model_types import DataType
from gmodel_lib.arch import *

def predict(transA, transB, m, n, batch_count, k, dtype, soc_s='gfx942', b_in_HBM=True, macroTiles=None):
    soc = None
    if soc_s == 'gfx942':
        from gmodel_lib.arch import MI300X
        soc = MI300X()
    else:
        print("SoC unavailable")
        assert(0)

    compute_dtype = None
    if dtype == 'torch.float16' or dtype == 'torch.bfloat16':
        compute_dtype = DataType.BF16
    elif dtype == 'torch.float8_e4m3fn' or dtype == 'torch.float8_e5m2':
        compute_dtype = DataType.FP8
    else:
        print("Datatype not found for GEMM")
        assert(0)

    p = GemmProblemFromSizes(m, n, k, batch_count, compute_dtype=compute_dtype)
    if b_in_HBM:
        p.force_to_cache(MemLoc.HBM, p.b)
    model = GemmModel(soc)
    sp = soc.default_solution_parms(p)

    tiles = []
    for mt in macroTiles:
        tiles.append(Tile(mt0=mt[0], mt1=mt[1], unroll=mt[2], split_summation=mt[3]))
    perf= model.simulate(p, model.make_solutions(p, solution_parms=sp, force_tiles=tiles))
    res = []
    for p in perf:
        res.append([
            p.solution.tile.mt0,
            p.solution.tile.mt1,
            p.solution.tile.unroll,
            p.solution.tile.split_summation
        ])
    return res
