################################################################################
#
# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

import yaml
import re
import argparse
import json
import copy
import os
import subprocess
import math
import numpy as np

# Paths to the input and output files
parser = argparse.ArgumentParser(description="""Generate Tensile config file""")

parser.add_argument(
    "--hipblaslt_log",
    type=str,
    help="Path to hipblaslt log file")

parser.add_argument(
    "--tensile_config", type=str,
    help="Path to tensile config file")

parser.add_argument(
    "--gpus", type=int, default=1,
    help="Number of gpus for tuning hipblaslt")

parser.add_argument(
    "--topk", type=int, default=None,
    help="Top k gemms for tuning")

parser.add_argument(
    "--iters", type=int, default=100,
    help="Max tuning iterations")

parser.add_argument(
    "--fast", type=bool, default=False,
    help="If enabled, only tune the matrix instruction with min tile sizes, else, tune full matrix instructions")

parser.add_argument(
    "--groups", type=bool, default=True,
    help="If enabled, will replace MatrixInstruction with GroupedMatrixInstruction")

parser.add_argument(
    "--gridbase_config", type=str, default=None,
    help="Range config path")

args = parser.parse_args()

NUM_WARM_UP = 20
ENQUEUES_PER_SYNC = 20
LibraryType = "GridBased"

CU_RE = r"Compute Unit:(?P<COMPUTE_UNIT>[\w ]+)"

res = subprocess.run("/opt/rocm/llvm/bin/offload-arch", stdout=subprocess.PIPE)
ArchitectureName = res.stdout.decode("utf-8").strip()
res = subprocess.run("rocminfo | grep Compute", stdout=subprocess.PIPE, shell=True, env={"ROCR_VISIBLE_DEVICES":"0"})
match = re.search(CU_RE, res.stdout.decode("utf-8").split('\n')[-2])
NUM_STAGES = 8
CU = 0
if match:
    CU = int(match.group('COMPUTE_UNIT').strip())
else:
    raise RuntimeError("Failed to get compute unit from rocminfo")

if ArchitectureName == 'gfx942':
    res = subprocess.run(["cat", "/sys/class/drm/card1/device/current_compute_partition"], stdout=subprocess.PIPE)
    if res.stdout.decode("utf-8").strip() == "CPX":
        XCC = 1
        GSU = [1,2,3,4,5,6,7,8]
    else:
        XCC = 4
        GSU = [1,2,3,4]
    DeviceNames = ["Device 0049", "Device 0050"]
    ScheduleName = "aquavanjaram"
elif ArchitectureName == 'gfx90a':
    XCC = 1
    GSU = [1,2,3,4]
    DeviceNames = ["Device 0050", "Device 0051", "Device 0052", "Device 0054", "Device 0062", "Device 7400", "Device 740c"]
    ScheduleName = "aldebaran"

fp16_instructions = [[16,16,16,1]]
bf16_instructions = [[16,16,8,1]]
tf32_instructions = [[16,16,8,1]]
fp32_instructions = [[16,16,4,1]]


HIPBLASLT_BENCH_RE = (
    r"(?P<CMD>\w+) --api_method c "
    r"-m (?P<M>[\d ]+)"
    r"-n (?P<N>[\d ]+)"
    r"-k (?P<K>[\d ]+)"
    r"--lda (?P<LDA>[\d ]+)"
    r"--ldb (?P<LDB>[\d ]+)"
    r"--ldc (?P<LDC>[\d ]+)"
    r"--ldd (?P<LDD>[\d ]+)"
    r"--stride_a (?P<STRIDE_A>[\d ]+)"
    r"--stride_b (?P<STRIDE_B>[\d ]+)"
    r"--stride_c (?P<STRIDE_C>[\d ]+)"
    r"--stride_d (?P<STRIDE_D>[\d ]+)"
    r"--alpha (?P<ALPHA>[\d\. ]+)"
    r"--beta (?P<BETA>[\d\. ]+)"
    r"--transA (?P<TRANS_A>[\w ]+)"
    r"--transB (?P<TRANS_B>[\w ]+)"
    r"--batch_count (?P<BATCH_COUNT>[\d ]+)"
    r"--a_type (?P<A_TYPE>[\w ]+)"
    r"--b_type (?P<B_TYPE>[\w ]+)"
    r"--c_type (?P<C_TYPE>[\w ]+)"
    r"--d_type (?P<D_TYPE>[\w ]+)"
    r"--scale_type (?P<SCALE_TYPE>[\w ]+)"
    r"--bias_type (?P<BIAS_TYPE>[\w ]+)"
    r"--compute_type (?P<COMPUTE_TYPE>[\w ]+)")


# Function to extract problem sizes from a line
def extract_problem_size(match):
    return [int(match.group('M').strip()), int(match.group('N').strip()), int(match.group('BATCH_COUNT').strip()), int(match.group('K').strip())]

def instruction_map(dtype_dict):
    if dtype_dict["DataType"] == 'S' and dtype_dict["F32XdlMathOp"] == 'x':
        return tf32_instructions
    elif dtype_dict["DataType"] == 'S' and dtype_dict["F32XdlMathOp"] == 0:
        return fp32_instructions
    elif dtype_dict["DataType"] == 'H':
        return fp16_instructions
    elif dtype_dict["DataType"] == 'B':
        return bf16_instructions
    else:
        return None

def datatype_map(dtype):
    if dtype == "f16_r":
        return "H"
    elif dtype == "f32_r":
        return "S"
    elif dtype == "xf32_r":
        return "XS"
    elif dtype == "bf16_r":
        return "B"
    else:
        return None

def trans_map(trans):
    if trans == "T":
        return True
    elif trans == "N":
        return False
    else:
        return None

def extract_dtype(match):
    DataType = datatype_map(match.group('A_TYPE').strip())
    DestDataType = datatype_map(match.group('C_TYPE').strip())
    ComputeDataType = datatype_map(match.group('COMPUTE_TYPE').strip())
    TransposeA = trans_map(match.group('TRANS_A').strip())
    TransposeB = trans_map(match.group('TRANS_B').strip())
    if DataType in ["H", "B"]:
        HighPrecisionAccumulate = True
    else:
        HighPrecisionAccumulate = False
    F32XdlMathOp = 0
    if ComputeDataType == "XS":
        ComputeDataType = "S"
        F32XdlMathOp = 'x'
    return {"Batched": True, "DataType": DataType, "DestDataType": DestDataType, "ComputeDataType": ComputeDataType, "TransposeA": TransposeA, "TransposeB": TransposeB, "HighPrecisionAccumulate": HighPrecisionAccumulate, "F32XdlMathOp": F32XdlMathOp, "OperationType": "GEMM", "UseBeta": True}

def find_matmul_instruction(mfma_instruction, size):
    for bm in range(int(math.log(mfma_instruction[3],2))+1):
        for m_tiles in reversed(range(1, CU+1)):
            m_tile_size = size[0] // m_tiles
            if m_tile_size > 256:
                continue
            wave_tile_m = math.ceil(m_tile_size / mfma_instruction[0])
            if wave_tile_m <= 0:
                continue
            for n_tiles in reversed(range(1, CU+1)):
                n_tile_size = size[1] // n_tiles
                if n_tile_size > 256:
                    continue
                wave_tile_n = math.ceil(n_tile_size / mfma_instruction[1])
                if wave_tile_n <= 0:
                    continue
                matmul_instruction = mfma_instruction + [2**bm, 1, 1, 1, 1]
                for k in reversed(range(3)):
                    if wave_tile_m // (2**k) >= 1 and wave_tile_m // (2**k) <= 32:
                        matmul_instruction[-4] = wave_tile_m // (2**k)
                        matmul_instruction[-2] = 2**k

                        for l in reversed(range(3)):
                            if wave_tile_n // (2**l) >= 1 and wave_tile_n // (2**l) <= 32:
                                matmul_instruction[-3] = wave_tile_n // (2**l)
                                matmul_instruction[-1] = 2**l

                                yield matmul_instruction

def get_groups(matmul_instruction_gen):
    # Extract skinny MTs for Groups
    NONTEMPORALRATIO = 8
    mi_groups0 = []
    mi_groups1 = []
    mi_left = []
    for mi in matmul_instruction_gen:
        if mi is not None:
            mt = [mi[0] * mi[5] * mi[7], mi[1] * mi[6] * mi[8]]
            ratio = mt[0] / mt[1]
            if ratio > NONTEMPORALRATIO:
                mi_groups0.append(mi)
            elif ratio < (1/NONTEMPORALRATIO):
                mi_groups1.append(mi)
            else:
                mi_left.append(mi)
    return mi_groups0, mi_groups1, mi_left

def extract_range(data):
    shapes = []
    if 'Exact' in data:
        shapes += [int(shape) for shape in data['Exact'].split(',')]
    if 'Range' in data:
        shape_range = data['Range'].split(':')
        points = data['Points']
        shapes += list(set(np.round(np.linspace(int(shape_range[0]), int(shape_range[1]), int(points))).astype(int).tolist()))
    return shapes

def split_gemms_by_gpus(unique_gemms, gpus):
    unique_gemms_subgroups = [None] * gpus
    for i, (k, v) in enumerate(unique_gemms.items()):
        if unique_gemms_subgroups[i%gpus] is not None:
            unique_gemms_subgroups[i%gpus].append((k, v))
        else:
            unique_gemms_subgroups[i%gpus] = [(k, v)]
    return unique_gemms_subgroups

def calculate_min_flops(m_sum, n_sum, batch_sum, k_sum, iters):
    m_avg = m_sum / len(unique_gemms_subgroup)
    n_avg = n_sum / len(unique_gemms_subgroup)
    batch_avg = batch_sum / len(unique_gemms_subgroup)
    k_avg = k_sum / len(unique_gemms_subgroup)

    return (ENQUEUES_PER_SYNC + args.iters) * m_avg * n_avg * batch_avg * k_avg / 2

def dump_yaml(gpu_idx, gemm_group, yaml_file, m_sum, n_sum, batch_sum, k_sum, iters, groups):
    MinFlopsPerSync = calculate_min_flops(m_sum, n_sum, batch_sum, k_sum, iters)
    # Read the YAML file
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    data["GlobalParameters"]["EnqueuesPerSync"] = ENQUEUES_PER_SYNC
    data["GlobalParameters"]["MaxEnqueuesPerSync"] = iters
    data["GlobalParameters"]["NumWarmups"] = NUM_WARM_UP
    data["GlobalParameters"]["MinFlopsPerSync"] = round(MinFlopsPerSync)

    # Update the ProblemSizes
    for i, dtype_str in enumerate(gemm_group):
        dtype = json.loads(dtype_str)

        if i >= len(data["BenchmarkProblems"]):
            data["BenchmarkProblems"].append(copy.deepcopy(data["BenchmarkProblems"][0]))
        data["BenchmarkProblems"][i][1]["BenchmarkFinalParameters"][0]["ProblemSizes"] = gemm_group[dtype_str]

        # Add groupd here if needed
        group_params = [[]]

        if groups:
            if dtype_str in groups:
                # Add Non Temporal
                groups[dtype_str][0]["NonTemporalB"] = [0, 4, 7]
                groups[dtype_str][1]["NonTemporalA"] = [0, 4, 7]
                for v in groups[dtype_str][0]["MatrixInstruction"].values():
                    for ntemp in groups[dtype_str][0]["NonTemporalB"]:
                        g = dict()
                        g["MatrixInstruction"] = list(v)
                        g["NonTemporalB"] = ntemp
                        group_params[0].append(g)
                for v in groups[dtype_str][1]["MatrixInstruction"].values():
                    for ntemp in groups[dtype_str][1]["NonTemporalA"]:
                        g = dict()
                        g["MatrixInstruction"] = list(v)
                        g["NonTemporalA"] = ntemp
                        group_params[0].append(g)
                for v in matmul_instructions.get(dtype_str, dict()).values():
                    g = dict()
                    g["MatrixInstruction"] = list(v)
                    group_params[0].append(g)
                for index, item in enumerate(data["BenchmarkProblems"][i][1]["ForkParameters"]):
                    if "MatrixInstruction" in item:
                        del item["MatrixInstruction"]
                        item["Groups"] = {}
            else:
                for index, item in enumerate(data["BenchmarkProblems"][i][1]["ForkParameters"]):
                    if "Groups" in item:
                        del item["Groups"]
                        item["MatrixInstruction"] = {}

        for item in data["BenchmarkProblems"][i][1]["ForkParameters"]:
            if ("Groups" in item) and group_params[0]:
                item["Groups"] = group_params
            elif "MatrixInstruction" in item:
                item["MatrixInstruction"] = [list(v) for v in matmul_instructions[dtype_str].values()]
            if "WorkGroupMappingXCCGroup" in item:
                item["WorkGroupMappingXCCGroup"] = [CU]
            if "WorkGroupMappingXCC" in item:
                item["WorkGroupMappingXCC"] = [XCC]
            if "GlobalSplitU" in item:
                item["GlobalSplitU"] = list(GSU)
        data["BenchmarkProblems"][i][0] = dtype
    data["LibraryLogic"]["DeviceNames"] = DeviceNames
    data["LibraryLogic"]["ScheduleName"] = ScheduleName
    data["LibraryLogic"]["ArchitectureName"] = ArchitectureName
    data["LibraryLogic"]["LibraryType"] = LibraryType
    # Write the updated YAML file
    yaml_file = os.path.basename(yaml_file)
    slices = yaml_file.split('.')
    with open(slices[0]+'.'+str(gpu_idx)+'.'+slices[1], 'w') as f:
        yaml.dump(data, f, default_flow_style=None)


if args.hipblaslt_log and args.gridbase_config is None:
    LibraryType = "Equality"
    unique_gemms = {}
    # Read problem sizes from the input file
    with open(args.hipblaslt_log, 'r') as f:
        for line in f:
            match = re.search(
                HIPBLASLT_BENCH_RE, line
            )
            if match:
                if line in unique_gemms:
                    unique_gemms[line] += 1
                else:
                    unique_gemms[line] = 1

    unique_gemms = {k: v for k, v in sorted(unique_gemms.items(), key=lambda item: item[1], reverse=True)[:args.topk]}
    for k, v in unique_gemms.items():
        print("Gemm config:", k, "Number:", v)

    unique_gemms_subgroups = split_gemms_by_gpus(unique_gemms, args.gpus)

    for gpu_idx, unique_gemms_subgroup in enumerate(unique_gemms_subgroups):
        gemm_group = {}
        matmul_instructions = {}
        groups = {}
        if unique_gemms_subgroup is None:
            continue

        m_sum = 0
        n_sum = 0
        batch_sum = 0
        k_sum = 0
        for k, v in unique_gemms_subgroup:
            match = re.search(
                HIPBLASLT_BENCH_RE, k
            )

            if match:
                size = extract_problem_size(match)
                original_size = copy.deepcopy(size)
                dtype = extract_dtype(match)
                mfma_instructions = instruction_map(dtype)
                dtype_str = json.dumps(dtype)
                if mfma_instructions is None:
                    continue
                mfma_instruction_found = False
                mfma_instruction = mfma_instructions[0]
                for _ in range(NUM_STAGES):
                    matmul_instruction_gen = list(find_matmul_instruction(mfma_instruction, size))
                    if args.groups:
                        mi_groups0, mi_groups1, matmul_instruction_gen = get_groups(matmul_instruction_gen)
                    else:
                        mi_groups0 = []
                        mi_groups1 = []

                    DIV_MI = 3 # 33.3%
                    MIN_MI = 5 # min 5 solutions
                        
                    total_inst = min(len(matmul_instruction_gen) // DIV_MI, MIN_MI)  # At least 5 insts and max of 33.3% of insts.
                    for index, matmul_instruction in enumerate(matmul_instruction_gen):
                        if matmul_instruction is not None:
                            if dtype_str not in matmul_instructions:
                                matmul_instructions[dtype_str] = dict()
                            matmul_instructions[dtype_str][str(matmul_instruction)] = matmul_instruction
                            if args.fast and (index > total_inst):
                                break
                    total_inst = min(len(mi_groups0) // DIV_MI, MIN_MI)
                    for index, mi_0 in enumerate(mi_groups0):
                        if dtype_str not in groups:
                            groups[dtype_str] = [{},{}]
                            groups[dtype_str][0]["MatrixInstruction"] = {}
                            groups[dtype_str][1]["MatrixInstruction"] = {}
                        groups[dtype_str][0]["MatrixInstruction"][str(mi_0)] = mi_0
                        if args.fast and (index > total_inst):
                            break
                    total_inst = min(len(mi_groups1) // DIV_MI, MIN_MI)
                    for index, mi_1 in enumerate(mi_groups1):
                        if dtype_str not in groups:
                            groups[dtype_str] = [{},{}]
                            groups[dtype_str][0]["MatrixInstruction"] = {}
                            groups[dtype_str][1]["MatrixInstruction"] = {}
                        groups[dtype_str][1]["MatrixInstruction"][str(mi_1)] = mi_1
                        if args.fast and (index > total_inst):
                            break
                    if len(matmul_instruction_gen) > 0 or len(mi_groups0) > 0 or len(mi_groups1) > 0:
                        mfma_instruction_found = True
                        break
                    else:
                        max_dim = int(np.argmax(size))
                        size[max_dim] = size[max_dim] // 2

                if not mfma_instruction_found:
                    print(f"Can't find mfma instructions for {original_size}, please contact hipblaslt expert")
                else:
                    if dtype_str in gemm_group:
                        gemm_group[dtype_str].append({'Exact': list(original_size)})
                    else:
                        gemm_group[dtype_str] = [{'Exact': list(original_size)}]
                    m_sum += original_size[0]
                    n_sum += original_size[1]
                    batch_sum += original_size[2]
                    k_sum += original_size[3]

        dump_yaml(gpu_idx, gemm_group, args.tensile_config, m_sum, n_sum, batch_sum, k_sum, args.iters, groups)

elif args.gridbase_config and args.hipblaslt_log is None:
    LibraryType = "GridBased"
    unique_gemms = {}
    gpus = args.gpus

    with open(args.gridbase_config, 'r') as f:
        datas = yaml.safe_load(f)
        for data in datas:
            m_shapes = extract_range(data['M'])
            n_shapes = extract_range(data['N'])
            batch_shapes = extract_range(data['Batch'])
            k_shapes = extract_range(data['K'])
            DataType = datatype_map(data['DataType'].strip())
            DestDataType = datatype_map(data['DestDataType'].strip())
            ComputeDataType = datatype_map(data['ComputeDataType'].strip())
            TransposeA = trans_map(data['TransposeA'])
            TransposeB = trans_map(data['TransposeB'])
            if DataType in ["H", "B"]:
                HighPrecisionAccumulate = True
            else:
                HighPrecisionAccumulate = False
            F32XdlMathOp = 0
            if ComputeDataType == "XS":
                ComputeDataType = "S"
                F32XdlMathOp = 'x'
            dtype = {"Batched": True, "DataType": DataType, "DestDataType": DestDataType, "ComputeDataType": ComputeDataType, "TransposeA": TransposeA, "TransposeB": TransposeB, "HighPrecisionAccumulate": HighPrecisionAccumulate, "F32XdlMathOp": F32XdlMathOp, "OperationType": "GEMM", "UseBeta": True, "UseBias": 1, "Activation": True, "ActivationType": "hipblaslt_all", "UseScaleAlphaVec": 1}
            dtype_str = json.dumps(dtype)
            for m in m_shapes:
                for n in n_shapes:
                    for batch in batch_shapes:
                        for k in k_shapes:
                            unique_gemms[(dtype_str,m,n,batch,k)] = [m,n,batch,k]

    unique_gemms_subgroups = split_gemms_by_gpus(unique_gemms, args.gpus)

    for gpu_idx, unique_gemms_subgroup in enumerate(unique_gemms_subgroups):
        gemm_group = {}
        matmul_instructions = {}
        m_sum = 0
        n_sum = 0
        batch_sum = 0
        k_sum = 0
        for k, size in unique_gemms_subgroup:
            size = list(size)
            original_size = copy.deepcopy(size)
            dtype_str = k[0]

            dtype = json.loads(dtype_str)
            mfma_instructions = instruction_map(dtype)
            if mfma_instructions is None:
                continue
            mfma_instruction_found = False
            mfma_instruction = mfma_instructions[0]
            for _ in range(NUM_STAGES):
                matmul_instruction_gen = list(find_matmul_instruction(mfma_instruction, size))
                total_inst = min(len(matmul_instruction_gen) // 3, 5)  # At least 5 insts and max of 33.3% of insts.
                for index, matmul_instruction in enumerate(matmul_instruction_gen):
                    if matmul_instruction is not None:
                        if dtype_str not in matmul_instructions:
                            matmul_instructions[dtype_str] = dict()
                        matmul_instructions[dtype_str][str(matmul_instruction)] = matmul_instruction
                        if args.fast and (index > total_inst):
                            break
                if len(matmul_instruction_gen) > 0:
                    mfma_instruction_found = True
                    break
                else:
                    max_dim = int(np.argmax(size))
                    size[max_dim] = size[max_dim] // 2
            if not mfma_instruction_found:
                print(f"Can't find mfma instructions for {original_size}, please contact hipblaslt expert")
            else:
                if dtype_str in gemm_group:
                    gemm_group[dtype_str].append({'Exact': list(original_size)})
                else:
                    gemm_group[dtype_str] = [{'Exact': list(original_size)}]
                m_sum += original_size[0]
                n_sum += original_size[1]
                batch_sum += original_size[2]
                k_sum += original_size[3]

        dump_yaml(gpu_idx, gemm_group, args.tensile_config, m_sum, n_sum, batch_sum, k_sum, args.iters, {})
